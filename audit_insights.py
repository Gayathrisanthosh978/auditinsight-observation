# Standard library imports
import json
import random
import re
import time
from datetime import date, datetime

# Third-party imports
import mysql.connector
import pandas as pd
import requests
import spacy
from dateutil import parser
from fuzzywuzzy import fuzz
from pandasql import sqldf

# Local application imports
from app.core.config import OLLAMA_BASE_URL, OLLAMA_MODEL
from app.core.logger import get_logger
from app.scenario_insights.utils import generate_sentence_vllm
logger = get_logger(__name__)

def generate_sentence(prompt):
    url = f"{OLLAMA_BASE_URL}/api/generate"
    data = {
        "model": OLLAMA_MODEL,
        "prompt": prompt
    }
    try:
        response = requests.post(url, json=data)
        res_list = response.content.split(b"\n")
        res_str = ""
        for k in res_list:
            if len(k) > 0:
                k_dict = json.loads(k.decode('utf-8'))
                if "response" in k_dict.keys():
                    res_str += k_dict['response']
        sentences = res_str.split('\n')
        output = ' '.join([item for item in sentences if not item.strip().startswith("Sure")])
        return output
    except Exception as e:
        logger.error(f"Error in generate_sentence: {e}")
        return ""

def partially_mask_columns(df, columns):
    for col in columns:
        # Check if column exists in dataframe before applying mask
        if col in df.columns:
            def mask_values(value):
                if isinstance(value, str):
                    return value[:4] + '*' * (len(value) - 4) if len(value) > 4 else value[:2] + '*' * (len(value) - 2)
                elif isinstance(value, int) or isinstance(value, float):
                    value_str = str(value)
                    return value_str[:4] + '*' * (len(value_str) - 4) if len(value_str) > 4 else value_str[:2] + '*' * (len(value_str) - 2)
                else:
                    return value
            df[col] = df[col].apply(mask_values)
        else:
            logger.warning(f"Column '{col}' not found in dataframe. Available columns: {list(df.columns)}")
    return df

def keywords(dependencies, data):
    data.columns = data.columns.str.lower()
    status_count = data['bnx_status'].value_counts(dropna=True)
    status_percentages = (status_count / len(data) * 100).round(1)
    status_keyword = f"name: {dependencies['Scenario'][0]}, "
    status_keyword += ", ".join([f"{value}: percentage {percentage}, count {count}" for value, percentage, count in zip(status_percentages.index, status_percentages.values, status_count.values)])
    calculation_logic = [', '.join([
        f"name: {dependencies['Scenario'][0]}",
        f"calculation logic: {dependencies['calculated_field_logic'][0]}",
        f"exception criteria: {dependencies['exception_criteria'][0]}"
    ])]
    number_of_records = {
        'Name': dependencies['Scenario'][0],
        'Records': data.shape[0]
    }
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

def feature_sel(data):
    """ Selecting important features using correlation method. """
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

def find_unique_and_missing_values(data):
    """ Finding unique and missing values (if any) from the selected features. """
    unique_counts = {col: data[col].nunique() for col in data}
    unique_counts_lessthan_total_greaterthan_one = {col: count for col, count in unique_counts.items() if count < len(data) and count > 1}
    sorted_columns = sorted(unique_counts_lessthan_total_greaterthan_one.items(), key=lambda x: x[1])
    selected_columns = [col for col, _ in sorted_columns[:2]]
    unique_missing_value__list = []
    for col in selected_columns:
        unique_values = data[col].nunique()
        all_values = data[col].unique().tolist()
        if unique_values == 2:
            selected_values = random.sample(all_values, 2)
        elif unique_values == 3:
            selected_values = random.sample(all_values, 3)
        else:
            selected_values = random.sample(all_values, 2)
        unique_values_dict = {
            'Name': col,
            'Unique Count': unique_values,
            'Selected Values': selected_values
        }
        unique_missing_value__list.append(unique_values_dict)
    missing_percentages = data.isnull().mean() * 100
    columns_having_missing_value = missing_percentages[(missing_percentages > 70) & (missing_percentages < 100)]
    missing_values_list = ', '.join([f'{col}: {percentage:.2f}' for col, percentage in columns_having_missing_value.items()])
    return unique_missing_value__list, missing_values_list

def replace_first_capital_with_lower(sentence):
    return re.sub(r'^([A-Z])', lambda x: x.group(1).lower(), sentence)

def parse_date(date_string):
    formats = ["%Y-%m-%d %H:%M:%S.%f%z", "%d-%m-%Y %H:%M:%S.%f%z", "%d-%m-%Y %H:%M:%S%z", "%d-%m-%Y"]
    for fmt in formats:
        try:
            parsed_date = datetime.strptime(date_string, fmt)
            return parsed_date
        except ValueError as err:
            logger.error(f"Error parsing date string '{date_string}' with format '{fmt}': {err}")
            continue
    logger.error("Unsupported date format")
    raise ValueError("Unsupported date format")

def generate_audit_sentence(date_created, current_date):
    audit_date_sentence = f"The audit was conducted between {date_created.strftime('%d-%m-%Y %H:%M:%S')} to {current_date.strftime('%d-%m-%Y %H:%M:%S')}."
    return audit_date_sentence

def run_queries(queries_out, df):
    results = []
    for query in queries_out:
        try:
            data_query = sqldf(query)
            results.append(data_query)
        except Exception as e:
            logger.error(f"Error running query: {query}\nError: {e}")
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
            logger.error("Error occurred in generate_audit_output, retrying...")
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
        prompt_audit = f"[INST] <<SYS>>\n{system_message_audit}\n<</SYS>>\nGive me audit observations for {sample_audit} scenario by considering the data sample {df}. {prompting_audit} Give only observations, no other characters are needed[/INST]"
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
        logger.info(f"Query results: {query_results}")
        content = [f"{query_name}: affected_records: {result['COUNT(*)'].values[0]}, Percentage: {round(((result['COUNT(*)'].values[0])/count_of_records)*100,2)}%, query: {out}" for query_name, result, out in zip(matches, query_results, queries_out)]
        logger.info("Query executed successfully")
        system_message_audit = f"Generate only 3 audit observations for the scenario {sample_audit}"
        prompting_audit = f"""Below content has information regarding audit observations that is to be formulated.
        Content to generate observations: {content}
        count_of_records: {count_of_records}
        control_names : {list(dependencies['control_name'])}
        Formatting Requirements: Present the findings in a numbered list, titled by the observation name followed by a full colon and the observation description (mention the specific detailing of columns in a well finished cooperate high level language). Also find appropriate control_name associated to each observation and provide it along with the impact, not as a separate sentence only if the control_name is suitable. Use the following format for the observations:
        1. Observation Name: On review of the scenario, [affected_records] out of [count_of_records] records had [mention the particular issue using the logic and columns used in the query]. This comes to [percentage]% of the population, potentially indicating [impact].
        2. Observation Name: Upon reviewing the scenario, it was found that [affected_records] out of [count_of_records] records had [mention the particular issue using the logic and columns used in the query]. This represents [percentage]% of the total records, suggesting significant [impact].
        3. Observation Name: An analysis of the scenario revealed that [affected_records] out of [count_of_records] records had [mention the particular issue using the logic and columns used in the query]. This amounts to [percentage]% of the total records, highlighting substantial [impact].
        Data Accuracy: Ensure the observations accurately reflect the content, avoiding any mention of non-existent issues like missing values for columns that are fully populated. No need to provide Note at the end.Also donot give control namesseperately. Use each value in content only once.
        Generate all 3 observations as per the above format using the observations from content. Do not give control names as separate give it along with impact as a sentence. Also no need of introductory statements like "Here are the three audit observations ...". if the observation ending in numbers add a space between the number and period.
        """
        prompt_audit = f"[INST] <<SYS>>\n{system_message_audit}\n<</SYS>>\n{prompting_audit} give me the output requested in the above format only in a numbered list, nothing else is needed as output[/INST]"
        out_audit = generate_sentence_vllm(prompt_audit)
    logger.info(f"Out audit: {out_audit}")
    return (out_audit, query_df, matches, queries_out, query_results, output_audit_sql)

def remove_words(sentence, words_to_remove):
    for word in words_to_remove:
        sentence = sentence.replace(word, "")
    return sentence

def similar(a, b):
    return fuzz.ratio(a.lower(), b.lower())

def identify_values(text, col_dict):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    identified_values = []
    for token in doc:
        match = None
        is_string = False
        for key, value in col_dict.items():
            value_str = str(value)
            if similar(value_str, token.text) > 60:
                match = key
                is_string = value_str.isalpha()
                break
        identified_values.append((token.text, match, is_string))
    return identified_values

def replace_date_with_empty_brackets(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == 'DATE':
            text = text.replace(ent.text, '{}')
    return text

def extract_substituted_words(output, formatted_output):
    formatted_output_escaped = re.escape(formatted_output).replace(r'\{\}', r'(.*?)')
    pattern = re.compile(formatted_output_escaped)
    match = pattern.match(output)
    if match:
        return match.groups()
    else:
        return ()

def try_parse_date(value):
    try:
        return parser.parse(value).date()
    except ValueError:
        logger.error(f"Error parsing date value: {value}")
        return None

def similarity(a, b, threshold):
    return fuzz.ratio(str(a).lower(), str(b).lower()) > threshold

def identify_val(values_identified, col_dict):
    identified_values = []
    for i in values_identified:
        match = None
        for threshold in range(100, 0, -1):
            for key, value in col_dict.items():
                if isinstance(value, date):
                    date_value = try_parse_date(i)
                    if date_value and similarity(date_value, value, threshold):
                        match = key
                        break
                elif similarity(value, i, threshold):
                    match = key
                    break
            if match:
                break
        identified_values.append((i, match))
    return identified_values

def generate_ai_insights(
        db_name,
        db_user,
        db_password,
        db_host,
        scenario_id,
        data_source_created_date,
        prev_generated_date
):
    cnx = mysql.connector.connect(
        user=db_user,
        password=db_password,
        host=db_host,
        database=db_name
    )
    try:
        dependencies = pd.read_sql(f'''select scenario.name as "Scenario",control.control_id,control.control_name,control.control_description,risk.risk_id,risk.risk_name,risk.risk_description,obsn.observation_id,obsn.observation as observation_name,obsn.description as observation_description,
                    scenario.calculated_field_logic, scenario.exception_criteria, scenario.selected_fields from audit_analytics_analyticsscenariomaster scenario
                    left join audit_analytics_analyticsscenariomaster_risk_control ctl on scenario.id=ctl.analyticsscenariomaster_id
                    left join risk_assessment_v1_riskassessmentdetails control on control.id =ctl.riskassessmentdetails_id
                    and control.deleted_at is null
                    left join (select distinct risk_id, control_id from risk_assessment_v1_controlrisk where deleted_at is null) rk on rk.control_id=control.id
                    left join risk_assessment_v2 risk on risk.id=rk.risk_id and risk.deleted_at is null
                    left join audit_reporting_observations_controls obn on obn.riskassessmentdetails_id=control.id
                    left join audit_reporting_observations obsn on obsn.id=obn.observation_id and obsn.deleted_at is null
                    where scenario.id={scenario_id};''', con=cnx)
        data = pd.read_sql(f''' SELECT * FROM {db_name}.analytics_builder_scenarioresults_{scenario_id};''', con=cnx)
        unique_keyword, missing_keyword = find_unique_and_missing_values(feature_sel(data))
        calculation_logic = ', '.join([
            f"{dependencies['calculated_field_logic'][0]}",
            f"exception criteria: {dependencies['exception_criteria'][0]}"
        ])
        columns_to_match = json.loads(dependencies["selected_fields"][0])
        columns_needed = re.findall(r'\[([^\]]*)\]', calculation_logic)
        hidden_columns = [column for column in columns_needed if column not in columns_to_match["fields"]]
        selected_columns = ['bnx_record_id'] + ['bnx_status'] + ['bnx_score'] + columns_needed
        try:
            data_with_calculationfields = data.loc[
                data['bnx_status'].isin(["Exception", "Exceptions"]),
                selected_columns
            ]
        except KeyError as e:
            logger.error(f"KeyError in data filtering: {e}")
            # Filter only existing columns
            existing_columns = [col for col in selected_columns if col in data.columns]
            data_with_calculationfields = data.loc[
                data['bnx_status'].isin(["Exception", "Exceptions"]),
                existing_columns
            ]
        if not hidden_columns:
            df = data_with_calculationfields
        else:
            df_new = data_with_calculationfields
            df = partially_mask_columns(df_new.copy(), hidden_columns)
        df = df.rename(columns={'bnx_score': 'Score'})
        row = data_with_calculationfields.iloc[0]
        row_dict = {col: value for col, value in row.items() if not col.startswith('bnx')}
        logger.info(f"Row dictionary: {row_dict}")
        sample_prompt = f"Scenario: {dependencies['Scenario'][0]}, Calculation Logic: {calculation_logic}, Calculation logic columns and values: {row_dict}, Status: Exceptions, Score: {row['bnx_score']}"
        sample_audit = dependencies['Scenario'][0]
        count_of_records = data.shape[0]
        data_exceptions = data.loc[data['bnx_status'].isin(["Exception", "Exceptions"])]
        exception_count = data_exceptions.shape[0]
        status_keyword, calculation_logic_kw, risk_control_audit_with_descriptions, number_of_records = keywords(dependencies, data)
        columns_needed = [item.lower() for item in columns_needed]
        Feature_columns = list(feature_sel(data).columns)
        merged_columns = list(set(columns_needed + Feature_columns))
        try:
            data_with_Feature_columns = data.loc[
                data['bnx_status'].isin(["Exception", "Exceptions"]),
                merged_columns
            ]
        except KeyError as e:
            logger.error(f"KeyError in feature columns filtering: {e}")
            # Filter only existing columns
            existing_feature_columns = [col for col in merged_columns if col in data.columns]
            data_with_Feature_columns = data.loc[
                data['bnx_status'].isin(["Exception", "Exceptions"]),
                existing_feature_columns
            ]
        if data_with_Feature_columns.shape[0] < 40000:
            data_with_Feature_columns = data_with_Feature_columns
        else:
            data_with_Feature_columns = data_with_Feature_columns.sample(40000)
        df1 = data_with_calculationfields
        df = data_with_Feature_columns
        sample_status = f"Keywords:[{status_keyword}]"
        system_msg_status = """Without introductory statements or heading, Directly provide a summary based on the input data without any introductory statements.Do not give sentence at begining like \"Based on the ....\" """
        prompting_status = """Explain the information in the keywords where name is the scenario name, exceptions and false positives are the categories of a column. Generate only one sentence based on the values from the keyword. The sentence must start with \"In the scenario\", explaining how many exceptions and false positives are there, with their percentage(must be in bracket) and count. Do not create any comments or greetings, do not hallucinate, and also do not give sentences which are not complete.Do not give scenario name explicitly:
        Refer the below example to get a better idea about the sentence needed:
        Examples:
          Keywords:['name: Missing Information, Open: percentage 66.7,count 10, Exceptions: percentage 33.3, count 5']
          Output:In the scenario Missing Information 10 instances (66.7%) are classified as Open while 5 instances (33.3%) are categorized as Exceptions.
          ###
          Keywords:['name: Overpayment_Exceptions, False Positive : percentage 45.0,count 11, Exception: percentage 30.0, count 8, Open: percentage 25.0, count 6']
          Output:Overpayment_Exceptions indicates that 11 cases (45.0%) are labeled as False Positives, 8 cases (30.0%) as Exceptions and the remaining 6 cases (25.0%) are categorized as open.
          ###
          Keywords:['name: Discrepancy between billing and lease charges, Exceptions: percentage 58.8, count 40, False Positive: percentage 41.2, count 38']
          Output:In the scenario Discrepancy between billing and lease charges, 40 instances (58.8%) are labeled as Exceptions and the remaining 38 instances (41.2%) as False Positive.
          ###
        """
        prompt_status = f"[INST] <<SYS>>\n{system_msg_status}\n<</SYS>>\n\n{prompting_status}Here is the input for generation:{sample_status}.[/INST]"
        sentence_status = generate_sentence_vllm(prompt_status)
        prompting_calculation_logic = """Explain the information in the keywords where name is the scenario name, calculation logic is the logic used to create the scenario. Generate only one sentence based on the values from the keyword. The sentence must explain what is causing the scenario and do not use calculation logic directly. Do not create any comments or greetings, do not hallucinate, and also do not give sentences which are not complete:
        Refer the below example to get a better idea about the sentence needed:
        Examples:
              Keywords:['name: Duplicate Customer Name, calculation logic: DUPLICATE([customer name id]), exception criteria: True']
              Output:It has been observed that the 'customer name id' is being duplicated in the data.
              ###
              Keywords:['name: Missing Customer TAX, calculation logic: ISNULL([VAT_NUMBER_]), exception criteria: 1']
              Output:In the data if the  VAT NUMBER doesnot have any value then it is taken as an exception .
              ###
              Keywords: ['name: Payment Analysis, calculation logic: IF([payment_amt]> 10,1,0), exception criteria: 1']
              Output:In the data, payment amount is greater than 10 which is causing the exception here.
              ###
              Keywords:["name: Duplicate Customer TAX, calculation logic: IF((([Missing_Tax] = 0) AND ([Dup_Tax] = 'True'})),1,0), exception criteria: 1"]
              Output:The scenario focuses on identifying cases where the Missing Tax value is zero and the Duplicate Tax value is True which is causing the exception here.
              ###
              Keywords:['name: Highest Amount, calculation logic: SUMD([ID],[AMT]),exception criteria: 0']
              Output:In the scenario, the sum is computed for amount by grouping the items based on their corresponding IDs.
              ###
              Keywords:['name: Supplier_Info, calculation logic: DISTINCT_ROW_FLAG([`First_Name`]),exception criteria: 1']
              Output:In the scenario, First_Name has got repeated values causing an exception here.
              ###
              Keywords:['name: Overpayment Exceptions, calculation logic: [total invoice_amount] <[total payment amount], exception criteria: 1']
              Output:In the data it can be seen that total invoice_amount is less than the total payment amount causing an exception here.
              ###
              Try to keep the same above format to generate sentence starting with In the scenario it has been observed that ... preceeding the explanation of logic without giving the logic formula directly
        """
        sample_calculation_logic = f"Keywords:{calculation_logic}"
        prompt_calculation_logic = f"[INST] <<SYS>>\n{system_msg_status}\n<</SYS>>\n\n{prompting_calculation_logic}Here is the input for generation:{sample_calculation_logic}.[/INST]"
        sentence_calculation_logic = generate_sentence_vllm(prompt_calculation_logic)
        summary_of_generated_exceptions = sentence_status + ' ' + sentence_calculation_logic
        prompting_status_characteristics = """Generate a single sentence from given keyword where name is the name of the scenario also the number of records is given.
        Refer the below example to get a better idea about the sentence needed:
        Example:
            Keyword:{'Name: Duplicate_Customer_Tax', 'Records: 180' }
            Output:The Duplicate_Customer_Tax scenario has 180 records.
            ###
        """
        prompt_characteristics = f"[INST] <<SYS>>\n{system_msg_status}\n<</SYS>>\n\n{prompting_status_characteristics}Here is the input for generation:{number_of_records}.[/INST]"
        characteristic_sentence = generate_sentence_vllm(prompt_characteristics)
        prompting_status_unique = """Without introductory statements, Explain the information in the keywords where name is the column name. Generate only one sentence related to the values from the keyword. The sentence must start with the name of columns, explaining how many unique count is given and list out the given values from selected values.Do not create any comments or greetings, do not hallucinate, and also do not give sentences which are not complete.Combine sentences using conjunctions if there are more than one.Do not show Keywords.Do not show sentence at starting like  \"Based on the input data given...\"
        Refer the below example to get a better idea about the sentence needed:
        Example:
          Keywords:{'Name': 'Customer_id', 'Unique Count': 2, 'Selected Values': ['596-AS', '637-RD']}
          Output:The Customer_id column has only 2 unique values, 596-AS and 637-RD, indicating that most cases are labeled either as 596-AS or 637-RD.
          ###
          Keywords:{'Name': 'Customer_Name', 'Unique Count': 10, 'Selected Values': ['ZEPHYR VISTA', 'SOLARA JUNCTION']}
          Output:The Customer_Name column contains a total of 10 unique values, including 'ZEPHYR VISTA', 'SOLARA JUNCTION' and others indicating the presence of duplicate values within the column.
          ###
          Create sentences as per the above examples.No need of any additional sentences.Do not show sentence at starting like  \"Based on the input data given...\".
        """
        unique_sentence = ""
        if unique_keyword:
            prompt_unique = f"[INST] <<SYS>>\n{system_msg_status}\n<</SYS>>\n\n{prompting_status_unique}Here is the input for generation:{unique_keyword}Try to keep the sentences different without changing the meaning.[/INST]"
            unique_sentence = generate_sentence_vllm(prompt_unique)
        prompting_status_missing = """Without introductory statements, Generate a single sentence regarding the information provided in the list of keywords along with their corresponding missing value percentages. Ensure that the sentence encapsulates the missing percentage for each values mentioned in the keywords list, rounding the values as demonstrated in the example. Aim to conclude the sentence similarly to the example provided but only at the end. Omit any unnecessary details, comments, or incomplete sentences.
        Refer the below example to get a better idea about the sentence needed:
        Example:
          Keywords:['Supplier_name: 14.56, Reg_number: 33.43']
          Output:The 'Supplier_name' has missing values exceeding 10% and 'Reg_number' has missing values exceeding 30%, indicating potential data incompleteness in these features.
          ###
        """
        if missing_keyword == '':
            dummy = ''
        else:
            prompt_missing = f"[INST] <<SYS>>\n{system_msg_status}\n<</SYS>>\n\n{prompting_status_missing} Here is the input for generation:{missing_keyword}.[/INST]"
            missing = generate_sentence_vllm(prompt_missing)
            missing_sentence = replace_first_capital_with_lower(missing)
            dummy = "On the other hand " + missing_sentence
        summary_of_data_sentences = characteristic_sentence + unique_sentence + dummy
        prompt_summary_of_data = f"[INST] <<SYS>>\nFine tune the given content in a coperate manner, remove sentence that make no sense if needed, give only the finetuned output\n<</SYS>>\n\nHere is the paragraph for finetuning:{summary_of_data_sentences}.Also no introductory statements like 'Here is a brief corporate-style summary:' is needed.[/INST]"
        summary_of_data = generate_sentence_vllm(prompt_summary_of_data)
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
        date_created = parse_date(data_source_created_date)
        current_date = datetime.now().strftime("%d-%m-%Y %H:%M:%S.%f+00:00")
        current_date_parsed = parse_date(current_date)
        if prev_generated_date is not None:
            last_run_date = parse_date(prev_generated_date)
            if current_date_parsed > last_run_date:
                date_created = last_run_date
            elif current_date_parsed == last_run_date:
                date_created = last_run_date
        audit_date_sentence = generate_audit_sentence(date_created, current_date_parsed)
        audit_sentence = f" The population considered contained {count_of_records} rows out of which {exception_count} exceptions were identified which is {round((exception_count / count_of_records) * 100, 2)}% of the population. "
        audit_data = filtered_data_audit(df, count_of_records, sample_audit, generate_audit_output, run_queries, dependencies)
        words_to_remove = ["Fine-tuned output:", "Here is the fine-tuned output:", "Here is a brief summary in corporate language:", "Here are three audit observations related to the scenario:", "Here is a short paragraph marizing the controls:", "Audit Observations", "Company Name", "**", "[END]", "Summary:", "<>", "[/INST]", "[SYS]", "[INST]", "<<SYS>>", "[</SYS>>]", "[</SYS>]", "</SYS>>", "[]", "[", "]", "</", ">/", "Observation name:", "Description:", "Audit Observation:", "Audit Observations:", "Recommendation:", "Observation:", "Recommendations:", "Certainly!", "<<SUMMARY>>", " [</SUMMARY>>]", "<<", ">>", "/", "SUMMARY", "SUM", "<", "INST", ">", "/", "'\'"]
        output_audit = remove_words(audit_data[0], words_to_remove)
        audit_summary = audit_date_sentence + audit_sentence + output_audit
        system_message_rowwise = """Craft a corporate summary that articulates the scenario's key details(try to use logic column values given), beginning with 'In the scenario...'. It should adeptly incorporate relevant values from the calculation logic columns given in sample prompt to contextualize the analysis without directly computing or emphasizing numerical discrepancies. The summary must encapsulate the situation, the applied logic for detection, and the implications or necessary actions, all within two sentences. Ensure clarity, professionalism, and the avoidance of repetitive or unnecessary information."""
        prompting_rowwise = """Generate a corporate summary starting with 'In the scenario...' and employing professional language to elucidate the key findings. While you may use values from the calculation logic columns for context, avoid using calculation logic fields DIRECTLY or stating numerical differences. Focus on describing the issue using the calculation logic provided,also the methodology for its identification, and its significance, all in a concise version of not more than two sentences, steering clear of repetition and calculations outside the scenario's described logic.Do not use any sentences from the prompting.Do not show any discrepancy.Show values that are given in calculation logic columns and values.Donot include any notes at the end."""
        prompt_rowwise = f"[INST] <<SYS>>\n{system_message_rowwise}\n<</SYS>>\n\n {sample_prompt} {prompting_rowwise} [/INST]"
        output_row_wise = generate_sentence_vllm(prompt_rowwise)
        words_to_remove1 = ["Here is a short paragraph summarizing the control:", "Here is a brief summary in corporate language:", "Here is a corporate summary that meets the requirements:", "Here is a summary of", "Based on the provided information, here is a summary of", "Company Name", "Here are 3 audit observations for the given data: ", "Let me know if this meets your expectations!", "Here is a corporate summary that articulates the key details of the scenario:", "**", "[END]", "Summary:", "<>", "[/INST]", "[SYS]", "[INST]", "<<SYS>>", "[</SYS>>]", "[</SYS>]", "</SYS>>", "[]", "[", "]", "Observation name:", "Description:", "Audit Observation:", "Audit Observations:", "Recommendation:", "Observation:", "Recommendations:", "Certainly!", "<<SUMMARY>>", " [</SUMMARY>>]", "<<", ">>", "/", "SUMMARY", "SUM", "<", "INST", ">", "/", "'\'", "SYS", "{"]
        output_row_wise = remove_words(output_row_wise, words_to_remove1)
        output_text_ref = output_row_wise
        row_dict.update({'Score': row['bnx_score']})
        extracted_values = identify_values(output_row_wise, row_dict)
        columns_identified = [tup for tup in extracted_values if None not in tup]
        for key, value in row_dict.items():
            key_str = str(key)
            value_str = str(value)
            if value_str in output_text_ref:
                output_text_ref = output_text_ref.replace(value_str, '{}')
        numeric_values = [value[0] for value in columns_identified if not value[2] and 'date' not in value[1].lower()]
        for i in numeric_values:
            output_text_ref = output_text_ref.replace(i, '{}')
        formatted_sentence = replace_date_with_empty_brackets(output_text_ref)
        values_identified = extract_substituted_words(output_row_wise, formatted_sentence)
        for key, value in row_dict.items():
            if isinstance(value, str):
                date_value = try_parse_date(value)
                if date_value is not None:
                    row_dict[key] = date_value
        extracted_values = identify_val(values_identified, row_dict)
        logger.info(f"Extracted values: {extracted_values}")
        columns_to_replace = [item[1] for item in extracted_values]
        logger.info(f"Columns to replace: {columns_to_replace}")
        df1['summary'] = df1.apply(
            lambda row: formatted_sentence.format(
                *[row.get(col, 'N/A') for col in columns_to_replace]
            ), 
            axis=1
        )
        summary_dict = dict(zip(df1['bnx_record_id'].astype(str), df1['summary']))
        sections = {
            "Summary of Generated Exceptions": summary_of_generated_exceptions,
            "Summary of Data": remove_words(summary_of_data, words_to_remove),
            "Risk": remove_words(output_risk, words_to_remove),
            "Control": remove_words(output_control, words_to_remove),
            "Observation": audit_summary,
            "exception_insights": summary_dict
        }
        Generative_ai_summary = {}
        for section_title, section_content in sections.items():
            if section_title == "exception_insights":
                Generative_ai_summary[section_title] = section_content
            elif section_content.strip():
                Generative_ai_summary[section_title] = section_content
        print("Completed")
        return Generative_ai_summary
    finally:
        cnx.close() 

def main():
    print("audit_insights main() called. Implement your workflow here.")

if __name__ == "__main__":
    main()
