import asyncio
import json
import os
import socket
import time
import uuid

import pandas as pd
from config.common_configs import options
from openai import OpenAI
from tqdm import tqdm

from src.prompts.analyst_prompt import get_assistant_prompt, get_data_prompt, get_goahead_for_assistant
from src.utils.Logger import Logger


class data_analyst:
    def __init__(self, datasources: list, schemas: list, **kwargs):
        """
        Initializes the DataInterpretation instance.

        Args:
            logger (Logger, optional): The logger instance. Defaults to None.
        """

        self.__logger = Logger().get_simple_logger("DataAnalysis")
        self.kwargs = kwargs
        if not os.path.exists(self.kwargs["tmp_dir"]):
            os.makedirs(self.kwargs["tmp_dir"])

        self.__model = kwargs["model"]
        self.__llm_instance = OpenAI(api_key=options["openai_api_key"])
        self.__assistant_reruns = kwargs["reruns"]
        self.__rerun = False
        self.__unique_run_id = str(uuid.uuid4())
        self.user_datasources = datasources
        self.user_schemas = schemas

        self.initialize_variables()

    def initialize_variables(self):
        self.assistant = None
        self.conversation = ""
        self.use_assistant = False
        self.filenames = []
        self.schemas = []
        self.file_ids = []
        self.filemap = {}

    def convert_to_csv(self, input_file, **kwargs):
        filename, file_extension = os.path.splitext(input_file)
        file_extension = file_extension.lower()
        output_file = os.path.join(self.kwargs["tmp_dir"], f"{filename}_{self.__unique_run_id}.csv")
        try:
            if file_extension in [".csv"]:
                # If the file is already a CSV, just save it as the output CSV
                df = pd.read_csv(input_file, low_memory=False)
                output_file = input_file
            elif file_extension in [".xlsx", ".xls"]:
                # Excel file
                df = pd.read_excel(input_file)
            elif file_extension in [".json"]:
                # JSON file
                with open(input_file) as f:
                    data = json.load(f)
                # Assuming the JSON can be directly converted to a DataFrame
                df = pd.json_normalize(data)
            elif file_extension in [".txt"]:
                # Plain text file (e.g., tab-delimited or other structured format)
                with open(input_file) as f:
                    lines = f.readlines()
                # Assuming tab-delimited text; adjust delimiter as needed
                df = pd.DataFrame(
                    [line.strip().split("\t") for line in lines[1:]],
                    columns=lines[0].strip().split("\t"),
                )
            else:
                print(f"Unsupported file format: {file_extension}")
                return None

            if len(df) == 0:
                self.__logger.info(f"Found no data in: {input_file}")
                return None
            else:
                # Save to CSV
                df.to_csv(output_file, index=False)
                self.__logger.info(f"File successfully processed")

        except Exception as e:
            self.__logger.exception(f"Error processing file: {e}")

        return output_file

    def create_assistant(self, **kwargs):
        assistant_prompt = get_assistant_prompt()
        self.assistant = self.__llm_instance.beta.assistants.create(
            instructions=assistant_prompt,
            model=self.__model,
            tools=[{"type": "code_interpreter"}],
            name=f"dashboard-llm-{self.__unique_run_id}",
        )
        self.assistant_id = self.assistant.id
        self.__logger.info(f"Created a new assistant with ID: {self.assistant_id}")

    def process_datasources(self, **kwargs):
        for file, schema in zip(self.user_datasources, self.user_schemas):
            converted_file = self.convert_to_csv(file)
            if converted_file is None:
                continue
            try:
                _ = json.load(open(schema))
                self.filenames.append(converted_file)
                self.filemap[converted_file] = file
                self.schemas.append(schema)
                try:
                    file = self.__llm_instance.files.create(file=open(converted_file, "rb"), purpose="assistants")
                    self.file_ids.append(file.id)
                    self.__logger.info(f"Upload File ID: {file.id}")
                except Exception as e:
                    self.__logger.info(f"File upload failed, Exception: {e}")
                    pass
            except Exception as e:
                self.__logger.info(f"Schema format not supported, use json. Exception: {e}")

    def create_thread(self, user_query: str, **kwargs):
        thread_ids, run_ids, user_prompts = [], [], []

        for i, (file, schema, fileid) in enumerate(zip(self.filenames, self.schemas, self.file_ids)):
            user_prompt = get_data_prompt(user_query, schema)

            # Create thread
            thread = self.__llm_instance.beta.threads.create(
                messages=[
                    {
                        "role": "user",
                        "attachments": [
                            {
                                "file_id": fileid,
                                "tools": [{"type": "code_interpreter"}],
                            }
                        ],
                        "content": user_prompt,
                    }
                ]
            )
            thread_ids.append(thread.id)

            run = self.__llm_instance.beta.threads.runs.create(thread_id=thread.id, assistant_id=self.assistant_id)
            run_ids.append(run.id)
            user_prompts.append(user_prompt)
            self.__logger.info(f"Data Interpretation Running for File: {os.path.basename(file)}")

        return thread_ids, run_ids

    def use_created_knowledge(self, query):
        system_prompt, user_prompt = get_goahead_for_assistant(query, self.conversation, self.schemas, self.filenames)
        response = (
            self.__llm_instance.chat.completions.create(
                model=self.__model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            .choices[0]
            .message.content
        )
        if "yes" in response.lower() or "use data analyst assistant" in response.lower():
            self.use_assistant = True
        self.conversation += f"""
user: {query}
agent: {response}
        """
        return response

    def sumarize_response(self, query, this_conversation):
        response = (
            self.__llm_instance.chat.completions.create(
                model=self.__model,
                messages=[
                    {
                        "role": "system",
                        "content": f"""
                # Conversation Context #
                {this_conversation}
                """,
                    },
                    {"role": "user", "content": query},
                ],
            )
            .choices[0]
            .message.content
        )
        return response

    def pool_response(self, thread_ids, run_ids, **kwargs):
        timeout = 1000  # Maximum timeout in seconds
        interval_time = 10  # Polling interval in seconds
        completed_files = 0  # Track number of completed files
        total_files = len(thread_ids)  # Total number of files

        # Initialize tqdm progress bar with total equal to the number of files
        with tqdm(total=total_files, desc="Processing Files", unit="file") as progress_bar:
            time_taken = 0  # Track elapsed time

            while time_taken < timeout:
                status = []
                completed_files_current = 0
                done_threads = []

                for threadid, runid, filename in zip(thread_ids, run_ids, self.filenames):
                    if threadid in done_threads:
                        continue
                    # Retrieve run status for each thread and log it
                    run = self.__llm_instance.beta.threads.runs.retrieve(thread_id=threadid, run_id=runid)
                    status.append(run.status)
                    self.__logger.info(f"Status for {os.path.basename(filename)}: {status[-1]}")

                    # Count the number of completed files
                    if run.status == "completed":
                        done_threads.append(threadid)
                        completed_files_current += 1

                # Update progress bar based on newly completed files
                progress_bar.update(completed_files_current)

                # Break if all files are completed
                if completed_files_current == total_files:
                    self.__logger.info("All files have been successfully processed.")
                    break

                # Break if any file has failed
                if "failed" in set(status):
                    self.__logger.error("One or more files failed. Trying again")
                    break

                # Wait for the next polling interval
                time.sleep(interval_time)
                time_taken += interval_time

            self.__logger.info(f"Data Interpretation took {time_taken} seconds")
            return status

    def destroy_assistant(self, **kwargs):
        try:
            assistant_id = self.assistant.id
            self.__llm_instance.beta.assistants.delete(assistant_id=self.assistant.id)
            self.__logger.info(f"Deleted Assistant Instance: {assistant_id}")
            self.assistant = None
        except:
            self.__logger.error(f"Failed to delete assistant instance: {self.assistant_id}")
        try:
            if type(self.file_ids) == list:
                for i, fileid in enumerate(reversed(self.file_ids)):
                    self.__llm_instance.files.delete(file_id=fileid)
                    self.__logger.info(f"Deleted File Instance: {fileid}")
                    self.file_ids.pop(len(self.file_ids) - 1 - i)
            else:
                self.__llm_instance.files.delete(file_id=self.file_ids)
                self.__logger.info(f"Deleted File Instance: {fileid}")
            print(f"File Ids left to delete: {self.file_ids}")
        except:
            self.__logger.error(f"Failed to delete file instance: {self.file_ids}")

        if not self.__rerun:
            for file in self.filenames:
                try:
                    if os.path.dirname(file) == self.kwargs["tmp_dir"]:
                        os.remove(file)
                        self.__logger.info(f"Deleted Temporary File: {file}")
                except:
                    continue

    def initialize_assistant(self):
        self.create_assistant()
        if len(self.file_ids) == 0 or len(self.filenames) < len(self.user_datasources):
            self.process_datasources()

    async def get_interpretations(self, *, user_query: str):
        """
        Retrieves the chart types for the given user query and data sources.

        Args:
            user_query (str): The user query.
            datasources (str): The data sources.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: The parsed result containing observations, reasoning, and final JSON.
        """
        this_conversation = f"""
user: {user_query}
        """
        rerun = 0
        status = "failed"
        try:
            while rerun < self.__assistant_reruns:
                try:
                    self.__logger.info(f"Starting data interpretation")
                    response = self.use_created_knowledge(query)
                    if self.use_assistant:
                        if self.assistant is None:
                            self.initialize_assistant()
                        thread_ids, run_ids = self.create_thread(user_query)
                        status = self.pool_response(thread_ids, run_ids)
                        if "failed" not in status:
                            self.__rerun = False
                            break
                        self.__rerun = True
                        self.destroy_assistant()
                        rerun += 1
                        self.__logger.warning("Assistant processing failed, retrying...")
                    else:
                        this_conversation += f"agent: `(using just schema)` {response}"
                except Exception as e:
                    self.__logger.exception(f"Error while running data interpretation: {e}")
                    self.destroy_assistant()
                    rerun += 1
                    if rerun < self.__assistant_reruns:
                        self.__logger.warning("Assistant processing failed, retrying...")
                    self.__rerun = True

            if self.use_assistant:
                task_results = {}
                for thread_id, filename in zip(thread_ids, self.filenames):
                    try:
                        assistant_response = (
                            self.__llm_instance.beta.threads.messages.list(thread_id=thread_id)
                            .data[0]
                            .content[0]
                            .text.value
                        )
                        if assistant_response is None:
                            continue

                        task_results[self.filemap[filename]] = assistant_response
                        this_conversation += f"""
data analyst assistant: (used file {os.path.basename(self.filemap[filename])})
{str(assistant_response)}
                        """
                    except Exception as e:
                        self.__logger.exception(f"Error in {filename} to process data: {e}")
                        continue

            response = self.sumarize_response(query, this_conversation)
            this_conversation += f"\nagent: {response}"
            self.conversation += f"""
user: {user_query}
agent: {response}
            """
            return this_conversation

        except Exception as e:
            self.__logger.exception(f"Encountered an exception: {e}")
            try:
                self.destroy_assistant()
            except Exception as e:
                self.__logger.exception(f"Error encountered: {e}")
            return

        except KeyboardInterrupt:
            return self.conversation


if __name__ == "__main__":
    from utils import main, render_markdown_to_pdf

    data, schema, output = main()
    if type(output) == list:
        output = output[0]
    else:
        str(output)

    try:
        host_ip = str(socket.gethostbyname(socket.gethostname()))
    except:
        host_ip = "127.0.0.1"

    args = {
        "tmp_dir": options["tmp_dir"],
        "openai_api_key": options["openai_api_key"],
        "model": options["openai_model_default"],
        "reruns": options["assistant_reruns"],
    }
    conversation = ""
    loop = asyncio.get_event_loop()
    agent = data_analyst(data, schema, **args)
    print("\n\n")
    try:
        while True:
            query = input("enter query: ")
            if query == "quit":
                raise KeyboardInterrupt
            response = loop.run_until_complete(agent.get_interpretations(user_query=query))
            print("*" * 100)
            conversation += f"""
# User: {query}
agent: {response}
            """
            print(response)
            print("*" * 100)
    except KeyboardInterrupt:
        render_markdown_to_pdf(conversation, output)
        print("\nKeyboardInterrupt detected. Cleaning up...")
        try:
            agent.destroy_assistant()
            print("Assistant destroyed successfully.")
        except Exception as e:
            print(f"Error during cleanup: {e}")
        finally:
            print("Exiting program.")
            exit(0)
