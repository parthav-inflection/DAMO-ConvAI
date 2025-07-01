#!/usr/bin/env python3
import argparse
import fnmatch
import json
import os
import pdb
import pickle
import re
import sqlite3
from typing import Dict, List, Tuple

import backoff
import openai
import pandas as pd
import sqlparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import threading
'''openai configure'''

# Configure API base URL for custom endpoints
openai.api_base = os.environ.get("API_BASE", "https://layercake.tacos.inf7ll8.com/model/inf-2-0-32b-sql/v1")

openai.debug=True


def new_directory(path):  
    if not os.path.exists(path):  
        os.makedirs(path)  


def get_db_schemas(bench_root: str, db_name: str) -> Dict[str, str]:
    """
    Read an sqlite file, and return the CREATE commands for each of the tables in the database.
    """
    asdf = 'database' if bench_root == 'spider' else 'databases'
    with sqlite3.connect(f'file:{bench_root}/{asdf}/{db_name}/{db_name}.sqlite?mode=ro', uri=True) as conn:
        # conn.text_factory = bytes
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        schemas = {}
        for table in tables:
            cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='{}';".format(table[0]))
            schemas[table[0]] = cursor.fetchone()[0]

        return schemas

def nice_look_table(column_names: list, values: list):
    rows = []
    # Determine the maximum width of each column
    widths = [max(len(str(value[i])) for value in values + [column_names]) for i in range(len(column_names))]

    # Print the column names
    header = ''.join(f'{column.rjust(width)} ' for column, width in zip(column_names, widths))
    # print(header)
    # Print the values
    for value in values:
        row = ''.join(f'{str(v).rjust(width)} ' for v, width in zip(value, widths))
        rows.append(row)
    rows = "\n".join(rows)
    final_output = header + '\n' + rows
    return final_output

def generate_schema_prompt(db_path, num_rows=None):
    # extract create ddls
    '''
    :param root_place:
    :param db_name:
    :return:
    '''
    full_schema_prompt_list = []
    conn = sqlite3.connect(db_path)
    # Create a cursor object
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    schemas = {}
    for table in tables:
        if table == 'sqlite_sequence':
            continue
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='{}';".format(table[0]))
        create_prompt = cursor.fetchone()[0]
        schemas[table[0]] = create_prompt
        if num_rows:
            cur_table = table[0]
            if cur_table in ['order', 'by', 'group']:
                cur_table = "`{}`".format(cur_table)

            cursor.execute("SELECT * FROM {} LIMIT {}".format(cur_table, num_rows))
            column_names = [description[0] for description in cursor.description]
            values = cursor.fetchall()
            rows_prompt = nice_look_table(column_names=column_names, values=values)
            verbose_prompt = "/* \n {} example rows: \n SELECT * FROM {} LIMIT {}; \n {} \n */".format(num_rows, cur_table, num_rows, rows_prompt)
            schemas[table[0]] = "{} \n {}".format(create_prompt, verbose_prompt)

    for k, v in schemas.items():
        full_schema_prompt_list.append(v)

    schema_prompt = "\n\n".join(full_schema_prompt_list)

    return schema_prompt

def generate_comment_prompt(question, knowledge=None):
    pattern_prompt_no_kg = "-- Using valid SQLite, answer the following questions for the tables provided above."
    pattern_prompt_kg = "-- Using valid SQLite and understading External Knowledge, answer the following questions for the tables provided above."
    # question_prompt = "-- {}".format(question) + '\n SELECT '
    question_prompt = "-- {}".format(question)
    knowledge_prompt = "-- External Knowledge: {}".format(knowledge)

    if not knowledge_prompt:
        result_prompt = pattern_prompt_no_kg + '\n' + question_prompt
    else:
        result_prompt = knowledge_prompt + '\n' + pattern_prompt_kg + '\n' + question_prompt

    return result_prompt

def cot_wizard():
    cot = "\nGenerate the SQL after thinking step by step: "
    
    return cot

def few_shot():
    ini_table = "CREATE TABLE singer\n(\n    singer_id         TEXT not null\n        primary key,\n    nation       TEXT  not null,\n    sname       TEXT null,\n    dname       TEXT null,\n    cname       TEXT null,\n    age    INTEGER         not null,\n    year  INTEGER          not null,\n    birth_year  INTEGER          null,\n    salary  REAL          null,\n    city TEXT          null,\n    phone_number   INTEGER          null,\n--     tax   REAL      null,\n)"
    ini_prompt = "-- External Knowledge: age = year - birth_year;\n-- Using valid SQLite and understading External Knowledge, answer the following questions for the tables provided above.\n-- How many singers in USA who is older than 27?\nThe final SQL is: Let's think step by step."
    ini_cot_result = "1. referring to external knowledge, we need to filter singers 'by year' - 'birth_year' > 27; 2. we should find out the singers of step 1 in which nation = 'US', 3. use COUNT() to count how many singers. Finally the SQL is: SELECT COUNT(*) FROM singer WHERE year - birth_year > 27;</s>"
    
    one_shot_demo = ini_table + '\n' + ini_prompt + '\n' + ini_cot_result
    
    return one_shot_demo

def few_shot_no_kg():
    ini_table = "CREATE TABLE singer\n(\n    singer_id         TEXT not null\n        primary key,\n    nation       TEXT  not null,\n    sname       TEXT null,\n    dname       TEXT null,\n    cname       TEXT null,\n    age    INTEGER         not null,\n    year  INTEGER          not null,\n    age  INTEGER          null,\n    salary  REAL          null,\n    city TEXT          null,\n    phone_number   INTEGER          null,\n--     tax   REAL      null,\n)"
    ini_prompt = "-- External Knowledge:\n-- Using valid SQLite and understading External Knowledge, answer the following questions for the tables provided above.\n-- How many singers in USA who is older than 27?\nThe final SQL is: Let's think step by step."
    ini_cot_result = "1. 'older than 27' refers to age > 27 in SQL; 2. we should find out the singers of step 1 in which nation = 'US', 3. use COUNT() to count how many singers. Finally the SQL is: SELECT COUNT(*) FROM singer WHERE age > 27;</s>"
    
    one_shot_demo = ini_table + '\n' + ini_prompt + '\n' + ini_cot_result
    
    return one_shot_demo



def extract_sql_from_response(response_text):
    """
    Extract SQL query from model response that may contain explanatory text.
    """
    import re
    
    # Remove the leading "SELECT" if it was prepended by our script
    if response_text.startswith('SELECT'):
        response_text = response_text[6:]  # Remove "SELECT"
    
    # Try to find SQL in code blocks first
    code_block_match = re.search(r'```(?:sql)?\s*(.*?)\s*```', response_text, re.DOTALL | re.IGNORECASE)
    if code_block_match:
        sql = code_block_match.group(1).strip()
        if sql.upper().startswith(('SELECT', 'INSERT', 'UPDATE', 'DELETE', 'WITH')):
            # Ensure SQL ends with semicolon
            if not sql.endswith(';'):
                sql += ';'
            return sql
    
    # Look for SQL keywords and extract everything from there to the end or semicolon
    sql_match = re.search(r'\b(SELECT|INSERT|UPDATE|DELETE|WITH)\b.*?(?:;|$)', response_text, re.DOTALL | re.IGNORECASE)
    if sql_match:
        sql = sql_match.group(0).strip()
        # Ensure SQL ends with semicolon
        if not sql.endswith(';'):
            sql += ';'
        return sql
    
    # If no SQL found, return the original response with semicolon if it looks like SQL
    response_text = response_text.strip()
    if response_text and not response_text.endswith(';') and any(keyword in response_text.upper() for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'WITH']):
        response_text += ';'
    return response_text

def generate_combined_prompts_one(db_path, question, knowledge=None):
    schema_prompt = generate_schema_prompt(db_path, num_rows=None) # This is the entry to collect values
    comment_prompt = generate_comment_prompt(question, knowledge)

    combined_prompts = schema_prompt + '\n\n' + comment_prompt + cot_wizard() + '\nSELECT '
    # combined_prompts = few_shot() + '\n\n' + schema_prompt + '\n\n' + comment_prompt

    # print(combined_prompts)

    return combined_prompts

def quota_giveup(e):
    return "quota" in str(e)

@backoff.on_exception(
    backoff.constant,
    Exception,
    giveup=quota_giveup,
    raise_on_giveup=True,
    interval=20
)
def connect_gpt(engine, prompt, max_tokens, temperature, stop):
    # print(prompt)
    try:
        # Convert to chat completion format
        messages = [{"role": "user", "content": prompt}]
        result = openai.ChatCompletion.create(
            model=engine, 
            messages=messages, 
            max_tokens=max_tokens, 
            temperature=temperature, 
            stop=stop
        )
    except Exception as e:
        result = 'error:{}'.format(e)
    return result
def process_single_question(args):
    '''Process a single question and return the result'''
    i, question, db_path, knowledge, api_key, engine = args
    
    # Set API key for this thread
    openai.api_key = api_key
    
    print(f'--------------------- processing {i}th question ---------------------')
    print(f'the question is: {question}')
    
    if knowledge:
        cur_prompt = generate_combined_prompts_one(db_path=db_path, question=question, knowledge=knowledge)
    else:
        cur_prompt = generate_combined_prompts_one(db_path=db_path, question=question)
    
    plain_result = connect_gpt(engine=engine, prompt=cur_prompt, max_tokens=4096, temperature=0, stop=None)
    
    if type(plain_result) == str:
        sql = plain_result
    else:
        raw_response = 'SELECT' + plain_result['choices'][0]['message']['content']
        sql = extract_sql_from_response(raw_response)
    
    db_id = db_path.split('/')[-1].split('.sqlite')[0]
    sql = sql + '\t----- bird -----\t' + db_id
    
    return i, sql

def collect_response_from_gpt(db_path_list, question_list, api_key, engine, knowledge_list=None, max_workers=5):
    '''
    :param db_path_list: list of database paths
    :param question_list: list of questions
    :param api_key: OpenAI API key
    :param engine: model engine name
    :param knowledge_list: list of knowledge (optional)
    :param max_workers: number of parallel workers (default: 5)
    :return: list of responses collected from openai
    '''
    # Prepare arguments for each question
    args_list = []
    for i, question in enumerate(question_list):
        knowledge = knowledge_list[i] if knowledge_list else None
        args = (i, question, db_path_list[i], knowledge, api_key, engine)
        args_list.append(args)
    
    # Process questions in parallel
    response_dict = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = [executor.submit(process_single_question, args) for args in args_list]
        
        # Collect results with progress bar
        for future in tqdm(futures, desc="Processing questions"):
            try:
                i, sql = future.result()
                response_dict[i] = sql
            except Exception as e:
                print(f"Error processing question: {e}")
                response_dict[len(response_dict)] = f"error: {e}"
    
    # Convert to ordered list
    response_list = [response_dict[i] for i in sorted(response_dict.keys())]
    
    return response_list

def question_package(data_json, knowledge=False):
    question_list = []
    for data in data_json:
        question_list.append(data['question'])

    return question_list

def knowledge_package(data_json, knowledge=False):
    knowledge_list = []
    for data in data_json:
        knowledge_list.append(data['evidence'])

    return knowledge_list

def decouple_question_schema(datasets, db_root_path):
    question_list = []
    db_path_list = []
    knowledge_list = []
    for i, data in enumerate(datasets):
        question_list.append(data['question'])
        cur_db_path = db_root_path + data['db_id'] + '/' + data['db_id'] +'.sqlite'
        db_path_list.append(cur_db_path)
        knowledge_list.append(data['evidence'])
    
    return question_list, db_path_list, knowledge_list

def generate_sql_file(sql_lst, output_path=None):
    result = {}
    for i, sql in enumerate(sql_lst):
        result[i] = sql
    
    if output_path:
        directory_path = os.path.dirname(output_path)  
        new_directory(directory_path)
        json.dump(result, open(output_path, 'w'), indent=4)
    
    return result    

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--eval_path', type=str, default='')
    args_parser.add_argument('--mode', type=str, default='dev')
    args_parser.add_argument('--test_path', type=str, default='')
    args_parser.add_argument('--use_knowledge', type=str, default='False')
    args_parser.add_argument('--db_root_path', type=str, default='')
    # args_parser.add_argument('--db_name', type=str, required=True)
    args_parser.add_argument('--api_key', type=str, required=True)
    args_parser.add_argument('--engine', type=str, required=True, default='code-davinci-002')
    args_parser.add_argument('--data_output_path', type=str)
    args_parser.add_argument('--chain_of_thought', type=str)
    args_parser.add_argument('--max_workers', type=int, default=5, help='Number of parallel workers for API requests (default: 5)')
    args = args_parser.parse_args()
    
    eval_data = json.load(open(args.eval_path, 'r'))
    # '''for debug'''
    # eval_data = eval_data[:3]
    # '''for debug'''
    
    question_list, db_path_list, knowledge_list = decouple_question_schema(datasets=eval_data, db_root_path=args.db_root_path)
    assert len(question_list) == len(db_path_list) == len(knowledge_list)
    
    if args.use_knowledge == 'True':
        responses = collect_response_from_gpt(db_path_list=db_path_list, question_list=question_list, api_key=args.api_key, engine=args.engine, knowledge_list=knowledge_list, max_workers=args.max_workers)
    else:
        responses = collect_response_from_gpt(db_path_list=db_path_list, question_list=question_list, api_key=args.api_key, engine=args.engine, knowledge_list=None, max_workers=args.max_workers)
    
    if args.chain_of_thought == 'True':
        output_name = args.data_output_path + 'predict_' + args.mode + '_cot.json'
    else:
        output_name = args.data_output_path + 'predict_' + args.mode + '.json'
    # pdb.set_trace()
    generate_sql_file(sql_lst=responses, output_path=output_name)

    print('successfully collect results from {} for {} evaluation; Use knowledge: {}; Use COT: {}'.format(args.engine, args.mode, args.use_knowledge, args.chain_of_thought))
