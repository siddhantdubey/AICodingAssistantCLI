import argparse
import os
import json
import termcolor
import openai

from dotenv import load_dotenv
from embed_project import (
    add_to_chromadb,
    process_code,
    read_code_from_file,
    client,
    get_embedding,
)

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

FUNCTIONS = [
    {
        "name": "search_codebase",
        "description": ("Semantically search the codebase for relevant methods and "
                        "get the 5 most relevant ones back. This is useful for when "
                        "the user is asking about workings of the codebase to either modify. "
                        "the codebase or learn more about the inner workings."
                        "Look stuff up more often than not unless it's basic python knowledge."),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A description/query of what you're looking for in the codebase."
                }
            }
        }
    }
]

def colored(text, color):
    return termcolor.colored(text, color)


def search_codebase(collection, query):
    return collection.query(
        query_embeddings=get_embedding(query),
        n_results=5
    )

def process_directory(collection, directory):
    readme_txt = None

    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(".py"):
                print(colored(f"Processing: {os.path.join(dirpath, filename)}", "cyan"))
                filepath = os.path.join(dirpath, filename)
                code = read_code_from_file(filepath)
                embeddings, sources = process_code(code, filename)
                if embeddings:
                    add_to_chromadb(collection, embeddings, sources)
            elif filename == "README.md" or filename == "README.rst":
                with open(os.path.join(dirpath, filename), 'r') as file:
                    readme_txt = file.read()

    return readme_txt


def handle_function_call(resp_function_call, collection, messages):
    arguments = resp_function_call["arguments"]
    query = json.loads(arguments)["query"]
    print(colored("Looking up code...", "yellow"))
    results = search_codebase(collection, query)
    list_of_pretty_results = [
        f'In {result_name}:\n {function_code}' for result_name, function_code in zip(results['ids'], results['documents'])
    ]
    string_results = f"Results: \n" + "\n".join(list_of_pretty_results)
    messages.append({
        "role": "user",
        "content": f"Here are 5 relevant code pieces from the codebase, formatted by file_name + class_name + method_name, when class name is available. Use these to construct your response to me, and make sure to list relevant code. \n {string_results}"
    })
    print(colored("Finishing up response...", "yellow"))
    response2 = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages
    )
    return response2["choices"][0]["message"]["content"], messages

def chat_interaction(collection, readme_txt):
    messages = [
        {
            "role":"system",
            "content": f"You are a helpful code assistant that will be answering questions about a very important large codebase. \
                        Before you answer a question from a user, make sure to lookup relevant context if you need it.\
                        {'This is the readme of the repo:' + readme_txt if readme_txt is not None else ''}"
        }
    ]
    print(colored("Welcome to code assistant, type quit to exit", "yellow"))
    while True:
        search_query = input(colored("\nUser: ", "green"))
        if search_query.lower() == 'quit':
            break
        messages.append({"role": "user", "content": search_query})
        print(colored("Thinking...", "yellow"))
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            functions=FUNCTIONS
        )
        resp_content = response["choices"][0]["message"]["content"]
        if resp_content:
            comb_response = resp_content
        else:
            resp_function_call = response["choices"][0]["message"]["function_call"]
            comb_response, messages = handle_function_call(resp_function_call, collection, messages)
        print(colored("Assistant: ", "yellow") + comb_response)
        messages.append({"role": "assistant", "content": comb_response})


def main():
    parser = argparse.ArgumentParser(description="Embed and perform semantic search on Python projects.")
    parser.add_argument('dir', metavar='DIR', type=str, help="Directory containing the Python project.")
    args = parser.parse_args()

    try:
        collection = client.get_collection(name="my_collection")
    except ValueError:
        collection = client.create_collection(name="my_collection")

    readme_txt = process_directory(collection, args.dir)
    chat_interaction(collection, readme_txt)


if __name__ == "__main__":
    main()
