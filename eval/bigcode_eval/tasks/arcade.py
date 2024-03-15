"""
Arcade: Natural Language to Code Generation in Interactive Computing Notebooks

https://https://arxiv.org/abs/2212.09248

Arcade is a collection of natural language to code problems on interactive data science notebooks. Each problem features an NL intent as problem specification, a reference code solution, and preceding notebook context (markdown or code cells). Arcade can be used to evaluate the accuracies of code large language models in generating data science programs given natural language instructions.
"""

import fcntl
import functools
import io
import itertools
import os
import pathlib
import shutil
import subprocess
import sys
import warnings
import zipfile

import requests
import tqdm

from bigcode_eval.base import Task

_CITATION = """
@article{yin22arcade,
    author = {Pengcheng Yin and Wen-Ding Li and Kefan Xiao and Abhishek Rao and Yeming Wen and Kensen Shi and Joshua Howland and Paige Bailey and Michele Catasta and Henryk Michalewski and Alex Polozov and Charles Sutton},
    title = {Natural Language to Code Generation in Interactive Data Science Notebooks},
    year = {2022},
    eprint = {arXiv:2212.09248},
}
"""


def create_all_tasks():
    def create_task(mode):
        class Arcade(GeneralArcade):
            def __init__(self):
                super().__init__(mode)
                
        return Arcade   
        
    return {
        f"arcade-{mode}": create_task(mode) for mode in ["base", "iosummary", "ioexample", "iotype"]
    }


class GeneralArcade(Task):
    DATASET_PATH = None
    DATASET_NAME = None

    def __init__(self, mode):
        super().__init__(
            stop_words=["# In[ ]:"], requires_execution=True
        )
        self.mode = mode
        self._dir = pathlib.Path(__file__).parent / "arcade_assets"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._src = self._dir / "arcade_nl2code"
        self._data_code = self._src / "annotated_dataset"
        self._data_root = self._data_code / "dataset/new_tasks/derived_datasets"
        if mode not in ["base", "iosummary", "ioexample", "iotype"]:
            raise ValueError("mode has to be either base, iosummary, ioexample or iotype")
        self.data_name = "arcade_{mode}.json"
        #self.data_name = "dataset.schema.originating_dfs.header_description.after_variable_cell.maxp2100.maxp_no_prefix-1.maxctxcell-1.json"
        #self.data_name = "dataset.schema.originating_dfs.header_description.after_variable_cell.maxp2100.maxp_no_prefix900.maxctxcell-1.e0_1_4_5.vanilla_prompting.json"
        self.artifacts_path = self._data_code / "dataset/new_tasks/artifacts"
        #self._download_source()
        #self._download_artifacts()

    #def _download_source(self):
    #    url = "https://github.com/HKUNLP/DS-1000/blob/49c1c543ada8b58138181333cdc62e613204efcf/ds1000.py?raw=true"
    #    lock = self._src.with_suffix(".lock")
    #    with open(lock, "w") as f_lock:
    #        fcntl.flock(f_lock, fcntl.LOCK_EX)
    #        if not self._src.exists():
    #            warnings.warn(f"DS-1000 source is being saved to {self._src}.")
    #            print("Downloading source code...")
    #            r = requests.get(url, stream=True)
    #            with open(self._src, "wb") as f_src:
    #                f_src.write(r.content)
    #            open(self._src.parent / "__init__.py", "w").close()
    #            print("Done.")
    #        fcntl.flock(f_lock, fcntl.LOCK_UN)

    def _download_source(self):
        url = "https://github.com/google-research/arcade-nl2code/archive/main.zip"
        zip_filename = self._src.parent / "arcade-nl2code.zip"
        folder_name_inside_zip = "arcade-nl2code-main/arcade_nl2code/"

        lock = self._src.with_suffix(".lock")
        with open(lock, "w") as f_lock:
            fcntl.flock(f_lock, fcntl.LOCK_EX)

            if not self._src.exists():
                warnings.warn(f"arcade-nl2code is being saved to {self._src}.")
                print("Downloading source code...")

                # Download the ZIP
                with requests.get(url, stream=True) as r, open(zip_filename, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

                # Extract only the desired folder
                with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
                    zip_ref.extractall(self._src.parent)

                # Move arcade_nl2code out of arcade-nl2code-main
                (self._src.parent / folder_name_inside_zip).rename(self._src)

                # Clean up the downloaded ZIP
                zip_filename.unlink()

                # Remove remaining files.
                shutil.rmtree(self._src.parent / "arcade-nl2code-main")
                print("Done.")

            fcntl.flock(f_lock, fcntl.LOCK_UN)

    def download_and_unzip_kaggle_dataset(self):
        # Set the desired paths
        download_path = self._data_code / "dataset"

        # Ensure the download directory exists
        download_path.mkdir(parents=True, exist_ok=True)

        # Use subprocess to run the kaggle command
        subprocess.run([
            'kaggle', 'datasets', 'download',
            '-d', 'googleai/arcade-nl2code-dataset',
            '-p', str(download_path)
        ], check=True)

        # Unzip the downloaded dataset
        zip_file = download_path / 'arcade-nl2code-dataset.zip'
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(download_path)

        # Delete the ZIP file after extraction if you want
        zip_file.unlink()

    def _download_artifacts(self):
        if self.artifacts_path.exists():
            print(f"{self.artifacts_path} exists. Skipping artifacts download.")
        else:
            self.download_and_unzip_kaggle_dataset()

            # Need to update a dataset version in this csv file.
            input_file = self._data_code / "dataset/new_tasks/kaggle_dataset_provenance.csv"
            temp_file = self._data_code / "dataset/new_tasks/temp.csv"
            with input_file.open("r") as infile, temp_file.open("w") as outfile:
                for index, line in enumerate(infile, start=1):
                    if index == 28:
                        parts = line.strip().split(",")
                        parts[-1] = "2"  # Modify the last column value
                        outfile.write(",".join(parts) + "\n")
                    else:
                        outfile.write(line)
            # Replace the original csv file with the modified file
            input_file.unlink()
            temp_file.rename(input_file)

            # Run the Python script with modified PYTHONPATH
            cmd = ["python", "build_new_tasks_split.py"]
            env = dict(PYTHONPATH=self._dir, **dict(os.environ))

            subprocess.run(cmd, cwd=self._data_code, env=env, check=True)

    def get_dataset(self):
        # TODO: retrieve the evaluation subset from the loaded dataset (e.g. `self.dataset["test"]`)
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        self.dataset = []
        sys.path.insert(0, str(self._dir))

        from arcade_nl2code.annotated_dataset import dataset as dataset_module

        flatten_dataset = []
        self.prompt_lens = dict()
        dataset = dataset_module.load_dataset(self._data_root / self.data_name)
        total = 0
        for episode in dataset:
            for turn_id, turn in enumerate(episode["turns"]):
                current_data = dict(
                    turn=turn,
                    turn_id=turn_id,
                    episode_metadata={k: v for k, v in episode.items() if k != "turns"}
                )
                flatten_dataset.append(current_data)
                self.prompt_lens[total] = len(turn.input)
                total += 1
        print(f"Creating {len(flatten_dataset)} testing examples.")
        return flatten_dataset

    def get_prompt(self, doc):
        """
        Builds the prompt for the LM to generate from.
        :param doc: dict[str: str]
            sample from the test dataset
        :return: str
        """
        return doc["turn"].input

    def get_reference(self, doc):
        # TODO: get the reference solution from a sample `doc` from the dataset
        """
        Builds the reference solution for the doc (sample from the test dataset).
        :param doc: dict[str: str]
            sample from the test dataset
        :return: str
        """
        return ""

    def postprocess_generation(self, generation, idx):
        """
        Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int (if needed)
            index of doc in the dataset to which the generation belongs
        :return: str
        """
        from arcade_nl2code.evaluation.processors import extract_first_cell_block
        return extract_first_cell_block(generation[self.prompt_lens[idx]:])

    def process_results(self, generations, references):
        # TODO: define how the evaluation score is computed from list of \
        # generations and reference solutions
        """
        Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations as in {"metric_name": result}.
        We encourage to directly load the metric from `evaluate` library to keep the code concise.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        :return: dict[str: float]
        """
        return {}

class GeneralDS1000(Task):
    DATASET_PATH = None
    DATASET_NAME = None

    def __init__(self, key, mode):
        super().__init__(
            stop_words=["</code>", "# SOLUTION END"], requires_execution=True
        )
        self._key = key
        self._mode = mode
        if self._key == "Matplotlib" and self._mode == "Insertion":
            warnings.warn("Insertion not supported for Matplotlib. Running Completion.")
            self._mode = "Completion"
        self._dir = pathlib.Path(__file__).parent / "ds"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._src = self._dir / "ds1000.py"
        self._data = self._dir / "ds1000_data"
        self._download_source()
        self._download_dataset()

    def _download_source(self):
        url = "https://github.com/HKUNLP/DS-1000/blob/49c1c543ada8b58138181333cdc62e613204efcf/ds1000.py?raw=true"
        lock = self._src.with_suffix(".lock")
        with open(lock, "w") as f_lock:
            fcntl.flock(f_lock, fcntl.LOCK_EX)
            if not self._src.exists():
                warnings.warn(f"DS-1000 source is being saved to {self._src}.")
                print("Downloading source code...")
                r = requests.get(url, stream=True)
                with open(self._src, "wb") as f_src:
                    f_src.write(r.content)
                open(self._src.parent / "__init__.py", "w").close()
                print("Done.")
            fcntl.flock(f_lock, fcntl.LOCK_UN)

    def _download_dataset(self):
        url = "https://github.com/HKUNLP/DS-1000/blob/49c1c543ada8b58138181333cdc62e613204efcf/ds1000_data.zip?raw=true"
        lock = self._data.with_suffix(".lock")
        with open(lock, "w") as f_lock:
            fcntl.flock(f_lock, fcntl.LOCK_EX)
            if not self._data.exists():
                warnings.warn(f"DS-1000 data is being saved to {self._data}.")
                print("Downloading dataset...")
                r = requests.get(url, stream=True)
                z = zipfile.ZipFile(io.BytesIO(r.content))
                z.extractall(self._dir)
                print("Done.")
            fcntl.flock(f_lock, fcntl.LOCK_UN)

    @functools.lru_cache()
    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        from .ds.ds1000 import DS1000Dataset

        data = DS1000Dataset(self._data, mode=self._mode).data
        if self._key == "All":
            if self._mode == "Insertion":
                warnings.warn(
                    "Insertion not supported for Matplotlib. Only running others."
                )
                data = {k: v for k, v in data.items() if k != "Matplotlib"}
            dataset = list(itertools.chain(*data.values()))
        else:
            dataset = data[self._key]
        return dataset

    def get_prompt(self, doc):
        """
        Builds the prompt for the LM to generate from.
        :param doc: dict[str: str]
            sample from the test dataset
        :return: str | dict[str: str]
        """
        if self._mode == "Completion":
            return doc["prompt"]
        elif self._mode == "Insertion":
            prefix, suffix = doc["prompt"].split("[insert]")
            prefix = f"{prefix.strip()}\n"
            suffix = f"\n{suffix.strip()}\n"
            return {"prefix": prefix, "suffix": suffix}
        else:
            raise ValueError(f"Invalid mode: {self._mode}")

    def get_reference(self, doc):
        """
        Builds the reference solution for the doc (sample from the test dataset).
        :param doc: dict[str: str]
            sample from the test dataset
        :return: str
        """
        return doc["reference_code"]

    def postprocess_generation(self, generation, idx):
        """
        Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int (if needed)
            index of doc in the dataset to which the generation belongs
        :return: str
        """
        if self._mode == "Completion":
            for start in ["BEGIN SOLUTION\n<code>", "# SOLUTION START"]:
                try:
                    generation = generation.split(start, 1)[-1]
                except IndexError:
                    pass
        for stop in self.stop_words:
            generation = generation.split(stop)[0]
        return generation.strip()

    def process_results(self, generations, references):
        """
        Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations as in {"metric_name": result}.
        We encourage to directly load the metric from `evaluate` library to keep the code concise.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        :return: dict[str: float]
        """
        dataset = self.get_dataset()
        num_correct = 0
        print("Scoring generations...")
        for i, ref in tqdm.tqdm(enumerate(references), total=len(references)):
            test = [doc for doc in dataset if doc["reference_code"] == ref][0]
            for gen in generations[i]:
                is_correct = test.test(gen)
                if is_correct:
                    num_correct += 1
        accuracy = num_correct / len(references) / len(generations[0])
        return {f"mean pass@1 accuracy ({len(generations[0])} samples)": accuracy}
