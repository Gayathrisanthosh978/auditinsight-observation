import json
import random
import re
import time
from datetime import datetime, date

import mysql.connector
import pandas as pd
import requests
import spacy
from bs4 import BeautifulSoup
from collections import Counter
from dateutil import parser
from fuzzywuzzy import fuzz
from pandasql import sqldf
import traceback

from app.core.logger import get_logger
from app.core.config import OLLAMA_BASE_URL, OLLAMA_MODEL
from app.scenario_insights.utils import generate_sentence_vllm

logger = get_logger(__name__)

def generate_audit(
    db_name,
    db_user,
    db_password,
    db_host,
    scenario_id,
    query,
    records_count,
    date_filters
):
    cnx = mysql.connector.connect(
        user=db_user,
        password=db_password,
        host=db_host,
        database=db_name
    )
    try:
        dependencies = pd.read_sql(
            f"""select scenario.name as "Scenario",control.control_id,control.control_name,control.control_description,
            risk.risk_id,risk.risk_name,risk.risk_description,obsn.observation_id,obsn.observation as observation_name,
            obsn.description as observation_description, scenario.calculated_field_logic, scenario.exception_criteria, 
            scenario.selected_fields from audit_analytics_analyticsscenariomaster scenario
            left join audit_analytics_analyticsscenariomaster_risk_control ctl on scenario.id=ctl.analyticsscenariomaster_id
            left join risk_assessment_v1_riskassessmentdetails control on control.id =ctl.riskassessmentdetails_id
            and control.deleted_at is null
            left join (select distinct risk_id, control_id from risk_assessment_v1_controlrisk where deleted_at is null) rk on rk.control_id=control.id
            left join risk_assessment_v2 risk on risk.id=rk.risk_id and risk.deleted_at is null
            left join audit_reporting_observations_controls obn on obn.riskassessmentdetails_id=control.id
            left join audit_reporting_observations obsn on obsn.id=obn.observation_id and obsn.deleted_at is null
            where scenario.id={scenario_id};""", con=cnx
        )

        filtered_data = pd.read_sql(query, con=cnx)
        data_1 = pd.read_sql(
            f"""SELECT * FROM {db_name}.analytics_builder_scenarioresults_{scenario_id};""", con=cnx
        )
        sample_audit = dependencies['Scenario'][0]
        calculation_logic = ', '.join([
            f"{dependencies['calculated_field_logic'][0].lower()}",
            f"exception criteria: {dependencies['exception_criteria'][0]}"
        ])

        def partially_mask_columns(df, columns):
            for col in columns:
                def mask_values(value):
                    if isinstance(value, str):
                        return value[:4] + '*' * (len(value) - 4) if len(value) > 4 else value[:2] + '*' * (len(value) - 2)
                    elif isinstance(value, int) or isinstance(value, float):
                        value_str = str(value)
                        return value_str[:4] + '*' * (len(value_str) - 4) if len(value_str) > 4 else value_str[:2] + '*' * (len(value_str) - 2)
                    else:
                        return value
                df[col] = df[col].apply(mask_values)
            return df

        def feature_sel(data):
            """Selecting important features using correlation method."""
            data.columns = data.columns.str.lower()
            data_copy = data.drop(columns=['score', 'status', 'exception flag', 'record id'], errors='ignore')
            data_copy = data_copy[data_copy['bnx_status'] != 'Open']
            columns_to_keep = [col for col in data_copy.columns if not (col.startswith('bnx') and col != 'bnx_status')]
            data_copy = data_copy[columns_to_keep]
            data_copy.fillna(0)
            encode_data = data_copy.copy()
            cat = encode_data.select_dtypes(include='object').columns
            for c in cat:
                encode_data[c] = encode_data[c].astype('category')
                encode_data[c] = encode_data[c].cat.codes
            corr_matrix = encode_data.corr().abs()
            corr_to_status = pd.DataFrame(corr_matrix.loc['bnx_status'])
            corr_to_status.reset_index(level=0, inplace=True)
            corr_to_status.rename(columns={'index': 'Feature'}, inplace=True)
            corr_to_status = corr_to_status[
                (corr_to_status['Feature'] != 'bnx_status')
            ].sort_values(by=['bnx_status'], ascending=False)
            len_columns = data_copy.shape[1]
            if len_columns < 10:
                high_corr_var = corr_to_status.iloc[:6]['Feature'].to_list()
            elif len_columns >= 20:
                len_columns_updated = int(len_columns * .2)
                high_corr_var = corr_to_status.iloc[:len_columns_updated]['Feature'].to_list()
            else:
                high_corr_var = corr_to_status.iloc[:8]['Feature'].to_list()
            return data[high_corr_var]

        def convert_to_list(date_filters):
            result = []
            if date_filters.get('month'):
                result.append(f"month: {date_filters['month']}")
            elif date_filters.get('quarter'):
                result.append(f"quarter: {date_filters['quarter']}")
            if date_filters.get('from_date'):
                result.append(f"from_date: {date_filters['from_date']}")
            if date_filters.get('to_date'):
                result.append(f"to_date: {date_filters['to_date']}")
            if date_filters.get('year'):
                result.append(f"year: {date_filters['year']}")
            return result

        date_list = convert_to_list(date_filters)

        def keywords(dependencies, data, records_count, date_filters):
            data.columns = data.columns.str.lower()
            status_count = data['bnx_status'].value_counts(dropna=True)
            status_percentages = (status_count / records_count * 100).round(3)
            status_keyword = f"Context/Scenario: {dependencies['Scenario'][0]}, date_filters: {date_filters}, Total Count: {records_count}, "
            status_keyword += ", ".join([
                f"{value}: percentage {percentage}, count {count}"
                for value, percentage, count in zip(status_percentages.index, status_percentages.values, status_count.values)
            ])
            calculation_logic = [', '.join([
                f"name: {dependencies['Scenario'][0]}",
                f"calculation logic: {dependencies['calculated_field_logic'][0]}",
                f"exception criteria: {dependencies['exception_criteria'][0]}"
            ])]
            number_of_records = data.shape[0]
            risk_with_description = {}
            control_with_description = {}
            audit_with_description = {}
            for index, row in dependencies.iterrows():
                risk_with_description[row["risk_name"]] = row["risk_description"]
                control_with_description[row["control_name"]] = row["control_description"]
                audit_with_description[row["observation_name"]] = row["observation_description"]
                audit_with_description = {key: value.replace('<p>', '').replace('</p>', '') if isinstance(value, str) else value for key, value in audit_with_description.items()}
            risk_control_audit_with_descriptions = [risk_with_description, control_with_description, audit_with_description]
            return (status_keyword, calculation_logic, risk_control_audit_with_descriptions, number_of_records)

        status_keyword, calculation_logic, risk_control_audit_with_descriptions, number_of_records = keywords(
            dependencies, filtered_data, records_count, date_list
        )

        system_msg_status = "Create a sentence that summarizes the following analysis:"
        prompting_status = f"""
            Context/Scenario: [Briefly describe the context or scenario being analyzed]
            date_filters: Specify the period during which the data was collected
            Total Count: [Provide the total number of items or events observed]
            Exceptions with percentage calculation: [Provide percentage of the specific outcome and the number in relation to the total count]
            Specific Outcome/Observation: [Describe the specific outcome or observation made from the scenario]
            Examples:
            ###
            Input:"Context/Scenario: product returns,date_filters: ["month: 1","from_date:2024-01-01", "to_date: 2024-01-31" , 'year': '2024'], Total Count: 2500, Exceptions: percentage 20, count 500"
            Output:"In the scenario of product returns, based on the analysis, the system observed that out of 2500 returns 500 (20%) were identified as defective items in January 2024."
            ###
            Input:"Context/Scenario: customer support tickets, date_filters: ["quarter: 2","from_date: 2024-04-01", "to_date: 2024-06-30", "year: 2023"], Total Count: 1800, Exceptions: percentage 33.333, count 300, False positive: percentage 16.66, count 300"
            Output:"In the scenario of customer support tickets, based on the analysis, the system observed that out of 1800 tickets, 300 (16.66%) were identified as false positives, and 600 (33.33%) were resolved within 24 hours in April-June 2023."
            ###
            Input:"Context/Scenario: product reviews, date_filters: ["from_date: 2024-06-01", "to_date: 2024-06-30", "year: 2024"], Total Count: 2500, Exceptions: percentage 20.0, count 500, False positive: percentage 10.0, count 250"
            Output:"In the scenario of product reviews, based on the analysis, the system observed that out of 2500 reviews, 250 (10.0%) were identified as false positives, and 500 (20.0%) were resolved within 24 hours in June 1 to June 30, 2024."
            ###
            Input to be generated:
            {status_keyword}
            Output:
            No introductory statement starting with "Here is the ..." is needed.
        """
        prompt_status = f"[INST] <<SYS>>\n{system_msg_status}\n<</SYS>>\n\n{prompting_status}Give only the Output:[/INST]"
        sentence_status = generate_sentence_vllm(prompt_status)
        summary_of_generated_exceptions = sentence_status

        risk_control_audit_with_descriptions_cleaned = [{k: v for k, v in d.items() if k is not None and v is not None} for d in risk_control_audit_with_descriptions]
        if risk_control_audit_with_descriptions_cleaned[0] != {}:
            system_msg_risk = "Generate a brief summary in corporate language based on the provided controls and their descriptions associated with the scenario."
            prompting_risk = f"The risks include: {risk_control_audit_with_descriptions_cleaned[0]} The summary should be limited to a single paragraph of no more than two sentences, directly reflecting the information given, without creating new content.Give risks as general not using words like 'our organisation' or 'the company'. Also no introductory statements like 'Here is a brief corporate-style summary:' is needed."
            prompt_risk = f"[INST] <<SYS>>\n{system_msg_risk}\n<</SYS>>\n\n{prompting_risk} Give only summary no other characters is needed. [/INST]"
            output_risk = generate_sentence_vllm(prompt_risk)
        else:
            output_risk = ""
        if risk_control_audit_with_descriptions_cleaned[1] != {}:
            system_msg_control = "Generate a brief summary in corporate language based on the provided controls and their descriptions associated with the scenario.."
            prompting_control = f"The controls include: {risk_control_audit_with_descriptions_cleaned[1]} The summary should be limited to a single paragraph of no more than two sentences, directly reflecting the information given, without creating new content.Give controls as general not using words like 'our organisation' or 'the company'. Also no introductory statements like 'Here is a brief corporate-style summary:' is needed."
            prompt_control = f"[INST] <<SYS>>\n{system_msg_control}\n<</SYS>>\n\n{prompting_control} Give only summary no other characters is needed[/INST]"
            output_control = generate_sentence_vllm(prompt_control)
        else:
            output_control = ""

        calculation_logic = ', '.join([
            f"{dependencies['calculated_field_logic'][0].lower()}",
            f"exception criteria: {dependencies['exception_criteria'][0]}"
        ])
        columns_to_match = json.loads(dependencies["selected_fields"][0])
        columns_to_match['fields'] = [column.lower() for column in columns_to_match['fields']]
        columns_needed = re.findall(r'\[([^\]]*)\]', calculation_logic)
        hidden_columns = [column for column in columns_needed if column not in columns_to_match["fields"]]
        if not hidden_columns:
            df = filtered_data
        else:
            df_new = filtered_data
            df = partially_mask_columns(df_new.copy(), hidden_columns)
        sample_audit = dependencies['Scenario'][0]
        count_of_records = df.shape[0]
        for name in df.columns:
            if name.startswith("po_number"):
                columns_needed.append(name)
        columns_needed = [item.lower() for item in columns_needed]
        Feature_columns = list(feature_sel(filtered_data).columns)
        merged_columns = list(set(columns_needed + Feature_columns))
        df = df.loc[:, ~df.columns.str.startswith('bnx')]
        df = df[merged_columns]

        df_sample = df
        if df_sample.applymap(lambda x: x == 'True' or x == 'False' or x is True or x is False).any().any():
            df_sample = df_sample.applymap(lambda x: 1 if x == 'True' or x is True else (0 if x == 'False' or x is False else x))

        def run_queries(queries_out, df):
            results = []
            for query in queries_out:
                try:
                    data_query = sqldf(query)
                    results.append(data_query)
                except Exception as e:
                    return None
            return results

        def generate_audit_output(sample_audit, df_sample):
            system_message_audit = """Generate 3 different audit observations related to the scenario for given data."""
            prompting_audit = f"""Generate 3 simple SQL queries related to the scenario without any GROUP BY based on the below instructions:
            Business Audit Observations: Analyze the below given data carefully and identify relevant issues based on the scenario {sample_audit}.
            Data that is to be analyzed: {df_sample}. The audit observations should be valid based on the data values.
            column names of the data: {list(df_sample.columns)}, use the column names and column values whether it is date also must be within double quotes in the query. Also start the query with SELECT 
            Features needed: Make sure that the logic of the 3 queries are different and imply 3 different specific issues of the given scenario, which are relevant and related to each other.
            Output needed: The table is named as df. I want 3 SQL queries that will help me identify relevant issues related to the scenario. Ensure that the queries result in a count.
            Output format that you are giving must be a numbered list refer the below to get clarity 
            1. observation name1 with short description : sql query
            2. observation name2 with short description : sql query
            3. observation name3 with short description : sql query
            Refer the below example to get a better idea about the sentence needed:
            Examples:
            1. Missing Customer Phone Numbers: SELECT COUNT(*) FROM  WHERE df "customer phone number" IS NULL;
            2. Missing Customer Addresses: SELECT COUNT(*) FROM df WHERE "customer address" IS NULL;
            3. Incomplete Customer Profiles where email,phone number or address is missing: SELECT COUNT(*) FROM customers WHERE "phone_number" IS NULL OR "email" IS NULL OR "address" IS NULL;
            4. Orders with Negative or Zero Quantity: SELECT COUNT(*) FROM df WHERE "quantity" <= 0;
            5. Products with Negative Price: SELECT COUNT(*) FROM df WHERE "price" < 0;
            Generate 3 different simple queries that are relevant to the given scenario. Include the column names used to make the observation more understandable and accurate. No queries with GROUP BY should be given. 
            If any specific values from data mentioned in the query, mention that also in the observation name.
            Make sure that the observations are high level issues as a company perspective.
            Make sure to provide valid SQL queries that can be run on the df dataframe with no background error.
            Ensure that the observations generated are different and related to the scenario.
            Give the 3 queries separated by a semicolon (;).
            """
            prompt_audit = f"[INST] <<SYS>>\n{system_message_audit}\n<</SYS>>\n{prompting_audit} Make sure that the sql queries is 100% accurate to run.Give the above mentioned output only no other unwanted characters or sentences is needed. Don't give 'Observation: ' or 'Description: ' .[/INST]"
            
            while True:
                output_audit = generate_sentence_vllm(prompt_audit, temperature=0.7, add_random_tag=True)
                if output_audit:
                    return output_audit
                else:
                    print("Error occurred, retrying...")
                    time.sleep(5)

        def filtered_data_audit(df_sample, records_count, sample_audit, generate_audit_output, run_queries, df):
            max_retries = 10
            retry_count = 0
            query_results = []

            while retry_count < max_retries:
                # Generate audit SQL output
                output_audit_sql = generate_audit_output(sample_audit, df_sample)
                
                # Extract SQL queries from the output
                queries = re.findall(r'\d+\.\s.*?:\s*(SELECT\s+.*?;)(?=\s*\d+\.|$)',output_audit_sql, re.IGNORECASE)
                queries_out = [re.sub(r'_(\s+)', '_', query.strip('`').strip()) for query in queries if query.strip('`').strip()]
                
                # Ensure at least 3 queries are extracted
                if len(queries_out) < 3:
                    retry_count += 1
                    continue

                # Check for duplicate queries
                if len(set(queries_out)) < len(queries_out):
                    retry_count += 1
                    continue

                # Run the extracted queries
                query_results = run_queries(queries_out, df)
                if query_results is None:
                    continue

                # Check if the query results are valid (no null values or zero count)
                valid_results = all(not df.empty and df.iloc[0, 0] > 0 for df in query_results)

                # To check the query results have multiple records dataframe in count*
                if any(result_df.shape[0] > 1 for result_df in query_results):
                    retry_count += 1
                    continue
                matches = re.findall(r'\d+\.\s(.*?):', output_audit_sql)
                if valid_results:
                    break
                retry_count += 1
            if retry_count == max_retries:
                logger.error("Failed to generate valid queries after maximum retries.")
                system_message_audit = "Generate 3 audit observations for given data."
                prompting_audit = """INSTRUCTIONS: 
                Business Audit Observations: Analyze the provided event in the sample_audit and identify relevant issues based on the data provided.
                Concise Descriptions: Each audit observation should be unique and succinct, limited to a maximum of two sentences without mentioning specific values.
                Formatting Requirements: Present the findings in a numbered list, titled by the observation name followed by a full colon and the observation description (Following is the Reference- observation name: observation description). List up to three observations.
                Data Accuracy: Ensure the observations accurately reflect the data, avoiding any mention of non-existent issues like missing values for columns that are fully populated.
                """
                prompt_audit = f"[INST] <<SYS>>\n{system_message_audit}\n<</SYS>>\nGive me audit observations for {sample_audit} scenario by considering the data sample {df_sample}. {prompting_audit} Give only observations, no other characters are needed[/INST]"
                out_audit = generate_sentence_vllm(prompt_audit)
                query_df = []
                matches = []
                queries_out = []
            else:
                logger.info("Successfully generated and ran queries.")
                queries_out_modified = [re.sub(r'SELECT\s+COUNT\(\*\)', 'SELECT *', query, flags=re.IGNORECASE) for query in queries_out]
                local_vars = {'df': df}
                query_df = [sqldf(query, local_vars) for query in queries_out_modified]
                if not (
                    isinstance(query_results, list) and 
                    all(isinstance(r, pd.DataFrame) for r in query_results)
                ):
                    query_results = []
                content = [f"{query_name}: affected_records: {result['COUNT(*)'].values[0]}, Percentage: {round(((result['COUNT(*)'].values[0])/records_count)*100,2)}%, query: {out}" for query_name, result, out in zip(matches, query_results, queries_out)]
                system_message_audit = f"Generate only 3 audit observations for the scenario {sample_audit}"
                prompting_audit = f"""Below content has information regarding audit observations that is to be formulated.
                Content to generate observations: {content}
                records_count: {records_count}
                control_names : {list(dependencies['control_name'])}
                Formatting Requirements: Present the findings in a numbered list, titled by the observation name followed by a full colon and the observation description (mention the specific detailing of columns in a well finished cooperate high level language). Also find appropriate control_name associated to each observation and provide it along with the impact, not as a separate sentence only if the control_name is suitable. Use the following format for the observations:
                1. Observation Name: On review of the scenario, [affected_records] out of [records_count] records had [mention the particular issue using the logic and columns used in the query]. This comes to [percentage]% of the population, potentially indicating [impact].
                2. Observation Name: Upon reviewing the scenario, it was found that [affected_records] out of [records_count] records had [mention the particular issue using the logic and columns used in the query]. This represents [percentage]% of the total records, suggesting significant [impact].
                3. Observation Name: An analysis of the scenario revealed that [affected_records] out of [records_count] records had [mention the particular issue using the logic and columns used in the query]. This amounts to [percentage]% of the total records, highlighting substantial [impact].
                Data Accuracy: Ensure the observations accurately reflect the content, avoiding any mention of non-existent issues like missing values for columns that are fully populated. No need to provide Note at the end.Also donot give control namesseperately. Use each value in content only once.
                Generate all 3 observations as per the above format using the observations from content. Do not give control names as separate give it along with impact as a sentence. Also no need of introductory statements like "Here are the three audit observations ...". if the observation ending in numbers add a space between the number and period.
                """
                prompt_audit = f"[INST] <<SYS>>\n{system_message_audit}\n<</SYS>>\n{prompting_audit} give me the output requested in the above format only in a numbered list, nothing else is needed as output[/INST]"
                out_audit = generate_sentence_vllm(prompt_audit)
            return (out_audit, query_df, matches, queries_out, query_results, output_audit_sql)

        def remove_words(sentence, words_to_remove):
            for word in words_to_remove:
                sentence = sentence.replace(word, "")
            return sentence

        words_to_remove = ["<", "[", "]", "INST", "Here is a brief summary in corporate language:", "Company Name", "**", "[END]", "Summary:", "<>", "[/INST]", "[SYS]", "[INST]", "<<SYS>>", "[</SYS>>]", "[</SYS>]", "</SYS>>", "[]", "[", "]", "Observation name:", "Description:", "Audit Observation:", "Audit Observations:", "Recommendation:", "Observation:", "Recommendations:", "Certainly!", "<<SUMMARY>>", " [</SUMMARY>>]", "<<", ">>", "/", "SUMMARY", "SUM", "<", "INST", ">", "/", "'\\'"]
        audit_data = filtered_data_audit(df_sample, records_count, sample_audit, generate_audit_output, run_queries, df)
        output_audit = remove_words(audit_data[0], words_to_remove)

        def df_to_list_of_dicts(df):
            return df.to_dict(orient='records')

        pattern_names = re.compile(r'\d+\.\s(.*?):')
        matches_title = pattern_names.findall(audit_data[0])
        converted_data = {audit_data[2][i]: df_to_list_of_dicts(df) for i, df in enumerate(audit_data[1])}
        categories_to_remove = [category for category, records in converted_data.items() if len(records) > 10]
        for category in categories_to_remove:
            del converted_data[category]

        def enclose_list_of_dicts_in_table_tag(data):
            if not data:
                return ""
            table_html = """
            <table border='1' style='font-family: Arial, sans-serif; margin: 10px 0; padding: 20px; background-color: #f0f0f0; border-collapse: collapse; width: 100%;'>
            """
            headers = "<tr>" + "".join(
                f"<th style='padding: 10px; border: 1px solid #ddd; text-align: left; max-width: 200px; word-wrap: break-word; overflow-wrap: break-word;'>{key}</th>"
                for key in data[0].keys()
            ) + "</tr>\n"
            table_html += headers
            for row in data:
                row_html = "<tr>" + "".join(
                    f"<td style='padding: 10px; border: 1px solid #ddd; text-align: left; max-width: 200px; word-wrap: break-word; overflow-wrap: break-word;'>{value}</td>"
                    for value in row.values()
                ) + "</tr>\n"
                table_html += row_html
            table_html += "</table>"
            return table_html

        html_tables = {name: enclose_list_of_dicts_in_table_tag(data) for name, data in converted_data.items()}

        def split_into_list(output_audit):
            pattern_output_audit = r'(1\.\s[^:]+:|2\.\s[^:]+:|3\.\s[^:]+:)'
            matches_output_audit = list(re.finditer(pattern_output_audit, output_audit))
            output_audit_list = []
            for i in range(len(matches_output_audit)):
                start = matches_output_audit[i].start()
                end = matches_output_audit[i+1].start() if i+1 < len(matches_output_audit) else len(output_audit)
                section = output_audit[start:end].strip()
                output_audit_list.append(section)
            return output_audit_list

        output_audit_list = split_into_list(output_audit)

        def count_html_rows(html_table):
            soup = BeautifulSoup(html_table, 'html.parser')
            rows = soup.find_all('tr')
            if rows:
                return len(rows) - 1
            return 0

        def extract_count_from_text(text):
            match = re.search(r'(\d+) out of \d+', text)
            if match:
                return int(match.group(1))
            return None

        def append_tables_to_audit(result_list, html_tables):
            updated_output_lines = []
            for line in result_list:
                expected_count = extract_count_from_text(line)
                if expected_count is not None:
                    matched_table = None
                    for key, html_table in html_tables.items():
                        table_row_count = count_html_rows(html_table)
                        if table_row_count == expected_count:
                            matched_table = html_table
                            break
                    if matched_table:
                        line += f"<br>{matched_table}"
                updated_output_lines.append(line)
            return '<br>'.join(updated_output_lines)

        result = append_tables_to_audit(output_audit_list, html_tables)

        def audit_date_sentence(sample_audit, date_list):
            system_prompt_intro = "Generate a statement regarding the dates of audit conducted using the query given"
            prompting_intro = f"""INSTRUCTIONS: 
            Create only one sentence for the scenario {sample_audit}.
            Mention only the information given in the list. 
            If a month is given in the list, mention the month only. If a quarter is given, mention the quarter only. If a date range is mentioned, give that date range.If any of the information is not available in the query, leave it out (do not hallucinate).
            Do not give incorrect information.
            Start with "The audit of the {sample_audit} scenario was conducted....". No need for sentences like "Here is the generated statement:".
            Do not replicate the information once added in the sentence.
            """
            prompt_intro = f"[INST] <<SYS>>\n{system_prompt_intro}\n<</SYS>>\n\ Given is the list to generate sentence {date_list}. {prompting_intro} Give only the sentence, no other extra sentence(Initial statement regarding the output or comments like a chatbot) is needed.[/INST]"
            output_intro = generate_sentence_vllm(prompt_intro)
            return output_intro

        audit_date = audit_date_sentence(sample_audit, date_list)

        sections = {
            "Summary of Observation": summary_of_generated_exceptions,
            "Detailed Observation": f"{audit_date}<br>The reasons for this could be:<br>{result}",
            "Control": remove_words(output_control, words_to_remove),
            "Risk": remove_words(output_risk, words_to_remove)
        }

        Generative_ai_summary = {}
        for section_title, section_content in sections.items():
            if section_title == "exception_insights":
                Generative_ai_summary[section_title] = section_content
            elif section_content.strip():
                Generative_ai_summary[section_title] = section_content

        description = ''
        for key, value in Generative_ai_summary.items():
            description += f'{key}: {value} '

        consolidated_data = {
            "consolidated": [
                {
                    "title": "Audit Observation",
                    "description": description.strip(),
                    "observed_date": datetime.now().strftime("%Y-%m-%d")
                }
            ],
            "transactional": []
        }

        json_string = json.dumps(consolidated_data, indent=4)
        parsed_data = json.loads(json_string)

        if parsed_data["consolidated"] == [{"title": "Audit Observation", 'description': description.strip(), 'observed_date': datetime.now().strftime("%Y-%m-%d")}]:
            Audit = parsed_data
        else:
            Audit = {}

        def format_description(description):
            if 'Summary of Observation:' in description:
                description = description.replace('Summary of Observation:', 'Summary of Observation:<br>')
            if 'Risk:' in description:
                description = description.replace('Risk:', '<br>Risk:<br>')
            if 'Control:' in description:
                description = description.replace('Control:', '<br>Control:<br>')
            if 'Detailed Observation:' in description:
                description = description.replace('Detailed Observation:', '<br> Detailed Observation:<br>')
            return description

        Audit['consolidated'][0]['description'] = format_description(Audit['consolidated'][0]['description'])

        def remove_newlines(data):
            if isinstance(data, dict):
                return {key: remove_newlines(value) for key, value in data.items()}
            elif isinstance(data, list):
                return [remove_newlines(item) for item in data]
            elif isinstance(data, str):
                return data.replace("\n", "")
            else:
                return data

        Audit_final = remove_newlines(Audit)
        return Audit_final
    finally:
        cnx.close()

