import pdb
from collections import defaultdict
from gridworld.tasks import Task, Tasks
from gridworld.data.load import download
import numpy as np
import pandas as pd
import os
import sys
import shutil
import json
from copy import deepcopy
from .utils import QuestionTask

from zipfile import ZipFile


BUILD_ZONE_SIZE_X = 11
BUILD_ZONE_SIZE_Z = 11
BUILD_ZONE_SIZE = 9, 11, 11

colors_to_hotbar = {
    "blue": 1,
    "green": 2,
    "red": 3,
    "orange": 4,
    "purple": 5,
    "yellow": 6,
}

if "IGLU_DATA_PATH" in os.environ:
    DATA_PREFIX = os.path.join(os.environ["IGLU_DATA_PATH"], "data", "cdm")
elif "HOME" in os.environ:
    DATA_PREFIX = os.path.join(os.environ["HOME"], ".iglu", "data", "iglu")
else:
    DATA_PREFIX = os.path.join(os.path.expanduser("~"), ".iglu", "data", "iglu")


class CDMDataset:
    """
    Dataset from paper Collaborative dialogue in Minecraft [1].

    Contains 156 structures of blocks, ~550 game sessions (several game sessions per
    structure), 15k utterances.

    Note that this dataset cannot split the collaboration into instructions since
    the invariant (of instruction/grid sequence) align does not hold for this dataset.


    [1] Anjali Narayan-Chen, Prashant Jayannavar, and Julia Hockenmaier. 2019.
    Collaborative Dialogue in Minecraft. In Proceedings of the 57th Annual Meeting
    of the Association for Computational Linguistics, pages 5405-5415, Florence,
    Italy. Association for Computational Linguistics.
    """

    ALL = {}
    DATASET_URL = {
        "v0.1.0-rc1": "https://iglumturkstorage.blob.core.windows.net/public-data/cdm_dataset.zip"
    }  # Dictionary holding dataset version to dataset URI mapping
    block_map = {
        "air": 0,
        "cwc_minecraft_blue_rn": 1,
        "cwc_minecraft_green_rn": 2,
        "cwc_minecraft_red_rn": 3,
        "cwc_minecraft_orange_rn": 4,
        "cwc_minecraft_purple_rn": 5,
        "cwc_minecraft_yellow_rn": 6,
    }

    def __init__(
        self, dataset_version="v0.1.0-rc1", task_kwargs=None, force_download=False
    ):
        """
        Dataset from paper Collaborative dialogue in Minecraft [1].

        Contains 156 structures of blocks, ~550 game sessions (several game sessions per
        structure), 15k utterances.

        Note that this dataset cannot split the collaboration into instructions since
        the invariant (of instruction/grid sequence) align does not hold for this dataset.


        [1] Anjali Narayan-Chen, Prashant Jayannavar, and Julia Hockenmaier. 2019.
        Collaborative Dialogue in Minecraft. In Proceedings of the 57th Annual Meeting
        of the Association for Computational Linguistics, pages 5405-5415, Florence,
        Italy. Association for Computational Linguistics.

        Args:
            dataset_version: Which dataset version to use.
            task_kwargs: Task-class specific kwargs. For reference see gridworld.task.Task class
            force_download: Whether to force dataset downloading
        """
        self.dataset_version = dataset_version
        self.task_index = None
        self.force_download = force_download
        if task_kwargs is None:
            task_kwargs = {}
        self._load_data(
            force_download=force_download,
        )
        self.task_kwargs = task_kwargs
        self.tasks = defaultdict(list)
        self.current = None
        for task_id, task_sessions in self.task_index.groupby("structure_id"):
            if len(task_sessions) == 0:
                continue
            for _, session in task_sessions.iterrows():
                task_path = os.path.join(
                    DATA_PREFIX, session.group, "logs", session.session_id
                )
                chat, target_grid = self._parse_task(task_path, task_id)
                task = Task(chat, target_grid, **self.task_kwargs)
                self.tasks[task_id.lower()].append(task)

    def reset(self):
        sample = np.random.choice(list(self.tasks.keys()))
        sess_id = np.random.choice(len(self.tasks[sample]))
        self.current = self.tasks[sample][sess_id]
        return self.current

    def __len__(self):
        return len(t for ts in self.tasks.values() for t in ts)

    def __iter__(self):
        for ts in self.tasks.values():
            for t in ts:
                yield t

    def set_task(self, task_id):
        self.current = self.tasks[task_id]
        return self.current

    def _load_data(self, force_download=False):
        path = sys.modules[__name__].__file__
        path_dir, _ = os.path.split(path)
        tasks = pd.read_csv(
            os.path.join(path_dir, "task_names.txt"),
            sep="\t",
            names=["task_id", "name"],
        )
        CDMDataset.ALL = dict(tasks.to_records(index=False))
        if not os.path.exists(DATA_PREFIX):
            os.makedirs(DATA_PREFIX, exist_ok=True)
        path = os.path.join(DATA_PREFIX, "data.zip")
        done = (
            len(list(filter(lambda x: x.startswith("data-"), os.listdir(DATA_PREFIX))))
            == 16
        )
        if done and not force_download:
            self.task_index = pd.read_csv(os.path.join(DATA_PREFIX, "index.csv"))
            shutil.rmtree(path, ignore_errors=True)
            return
        if force_download:
            for dir_ in os.listdir(DATA_PREFIX):
                if dir_.startswith("data-"):
                    shutil.rmtree(os.path.join(DATA_PREFIX, dir_), ignore_errors=True)
        if not os.path.exists(path) or force_download:
            download(
                url=CDMDataset.DATASET_URL[self.dataset_version],
                destination=path,
                data_prefix=DATA_PREFIX,
            )
            with ZipFile(path) as zfile:
                zfile.extractall(DATA_PREFIX)
        self.task_index = pd.read_csv(os.path.join(DATA_PREFIX, "index.csv"))
        shutil.rmtree(path, ignore_errors=True)

    def _parse_task(self, path, task_id, update_task_dict=False):
        if not os.path.exists(path):
            # try to unzip logs.zip
            path_prefix, top = path, ""
            while top != "logs":
                path_prefix, top = os.path.split(path_prefix)
            with ZipFile(os.path.join(path_prefix, "logs.zip")) as zfile:
                zfile.extractall(path_prefix)
        with open(os.path.join(path, "postprocessed-observations.json"), "r") as f:
            data = json.load(f)
        data = data["WorldStates"][-1]
        chat = "\n".join(data["ChatHistory"])
        target_grid = np.zeros(BUILD_ZONE_SIZE, dtype=np.int32)
        total_blocks = 0
        for block in data["BlocksInGrid"]:
            coord = block["AbsoluteCoordinates"]
            x, y, z = coord["X"], coord["Y"], coord["Z"]
            if not (-5 <= x <= 5 and -5 <= z <= 5 and 0 <= y <= 8):
                continue
            target_grid[
                coord["Y"] - 1, coord["X"] + 5, coord["Z"] + 5
            ] = CDMDataset.block_map[block["Type"]]
            total_blocks += 1
        if update_task_dict:
            colors = len(np.unique([b["Type"] for b in data["BlocksInGrid"]]))
            CDMDataset.ALL[
                task_id
            ] = f"{CDMDataset.ALL[task_id]} ({total_blocks} blocks, {colors} colors)"
        return chat, target_grid

    def __repr__(self):
        tasks = ", ".join(f'"{t}"' for t in self.tasks.keys())
        return f"TaskSet({tasks})"

    @staticmethod
    def subset(task_set):
        return {k: v for k, v in CDMDataset.ALL.items() if k in task_set}


class CDMQDataset(CDMDataset):
    """
    This contains the same datapoints as CDMDataset but also includes the dialog turns where q question was asked.
    """

    def __init__(self, split=None):

        self.path = os.path.join(
            os.path.expanduser("~"),
            ".iglu",
            "data",
        )

        self.questions = self._read_questions()
        self.tasks = defaultdict(list)
        path = sys.modules[__name__].__file__
        path_dir, _ = os.path.split(path)
        splits_file = os.path.join(path_dir, "resources/splits.json")
        with open(splits_file) as f:
            self.splits = json.load(f)
        self.split = split
        self.logs = self.create_logs()
        self.parse_logs()

    def _read_questions(self):
        path = sys.modules[__name__].__file__
        path_dir, _ = os.path.split(path)
        questions = []
        questions_file = os.path.join(
            path_dir, "resources/builder_utterance_labels.json"
        )
        with open(questions_file) as f:
            utterances = json.load(f)
        for id, list_of_utts in utterances.items():
            for q, label in list_of_utts:
                if label == "Instruction-level Questions":
                    questions.append(q)
        return set(questions)

    def create_logs(self):
        folders = [x for x in sorted(os.listdir(self.path)) if x.startswith("data")]
        logs = {}
        architect_line = ""
        builder_line = ""
        m = len("<Architect>")
        b = len("<Builder>")
        for folder in folders:
            with open(f"{self.path}/{folder}/dialogue-with-actions.txt") as f_d:
                dialogue = f_d.readlines()

            for line in dialogue:
                line = line.strip()
                line = line.strip("\n")
                if line in os.listdir(f"{self.path}/{folder}/logs"):
                    event = line
                    logs[event] = ""
                elif (line.lower() in self.questions) and builder_line:
                    builder_line += line[b:]
                elif line.lower() in self.questions:
                    if architect_line:
                        logs[event] += architect_line + "\n"
                        architect_line = ""
                    builder_line = line
                elif line.startswith("<Builder>"):
                    continue
                elif line.startswith("<Architect>") and architect_line:
                    architect_line += line[m:]
                    if not architect_line[-1] == ".":
                        architect_line += "."
                elif line.startswith("<Architect>"):
                    if builder_line:
                        logs[event] += builder_line + "\n"
                        builder_line = ""
                    architect_line = line
                    if not architect_line[-1] == ".":
                        architect_line += "."
                else:
                    if architect_line:
                        logs[event] += architect_line + "\n"
                    logs[event] += line + "\n"
                    architect_line = ""
        return logs

    def create_task(
        self, previous_chat, initial_grid, target_grid, last_instruction, target_label
    ):
        starting_grid = Tasks.to_sparse(initial_grid)
        utts = previous_chat.split("\n")
        """
        the target label is 1 for clear dialog turns 
        and 0 for turns followed by questions
        """
        if target_label == False:
            if len(utts) > 1:
                chat = "\n".join(utts[:-1])
                last_instruction = utts[-2]
            else:
                chat = previous_chat
                last_instruction = utts[-1]

        else:
            chat = previous_chat
            last_instruction = utts[-1]

        task = QuestionTask(
            chat=chat,
            target_grid=Tasks.to_dense(target_grid),
            starting_grid=starting_grid,
            last_instruction=last_instruction,
            is_question=target_label,
        )
        # To properly init max_int and prev_grid_size fields
        task.reset()
        return task

    def parse_logs(self):
        x_orientation = ["left", "right"]
        y_orientation = ["lower", "upper"]
        z_orientation = ["before", "after"]
        for key in self.logs:
            task_id = key.split("-")[2]
            if task_id not in self.splits[self.split]:
                continue
            history = self.logs[key]
            new_history = ""

            x_0, y_0, z_0 = 0, 0, 0
            first_block = False
            prev_line_builder = False
            prev_line_question = False
            built_grid = []
            target_grid = []

            for line in history.split("\n")[:-1]:
                line = line.strip()
                if line and line.startswith("<B"):
                    task = self.create_task(
                        new_history,
                        deepcopy(built_grid),
                        deepcopy(target_grid),
                        line,
                        True,
                    )
                    self.tasks[task_id].append(task)
                    prev_line_question = True
                    prev_line_builder = False
                    new_history += line + "\n"
                elif line and line.startswith("<A"):
                    task = self.create_task(
                        new_history,
                        deepcopy(built_grid),
                        deepcopy(target_grid),
                        "",
                        False,
                    )
                    self.tasks[task_id].append(task)
                    if prev_line_builder or prev_line_question:
                        new_history += "\n"
                    new_history += line + "\n"
                    built_grid = deepcopy(target_grid)
                    prev_line_builder = False
                    prev_line_question = False
                elif line:
                    if not prev_line_builder:
                        new_history += "<Builder> "
                    else:
                        new_history += " "
                    instruction = line.split("Builder ")[1]
                    command, info = instruction.split(" a ")
                    colour, coords = info.split(" block at")
                    command = "put" if command == "puts down" else "pick"

                    x_orig, y_orig, z_orig = [m.split(":")[1] for m in coords.split()]
                    x_orig, y_orig, z_orig = int(x_orig), int(y_orig), int(z_orig[:-1])
                    if not (
                        -5 <= x_orig <= 5 and -5 <= z_orig <= 5 and 0 <= y_orig < 8
                    ):
                        continue
                    # create grid tuple representation:
                    target_grid.append(
                        (x_orig, y_orig, z_orig, colors_to_hotbar[colour])
                    )
                    x = x_orig - x_0
                    y = y_orig - y_0
                    z = z_orig - z_0
                    if first_block:
                        if x != 0 and y != 0 and z != 0:
                            order = f"{command} {colour} {abs(x)} {x_orientation[int(x > 0)]}, {abs(y)} {y_orientation[int(y > 0)]} and {abs(z)} {z_orientation[int(z > 0)]}."
                        elif x == 0 and y != 0 and z != 0:
                            order = f"{command} {colour} {abs(y)} {y_orientation[int(y > 0)]} and {abs(z)} {z_orientation[int(z > 0)]}."
                        elif y == 0 and x != 0 and z != 0:
                            order = f"{command} {colour} {abs(x)} {x_orientation[int(x > 0)]} and {abs(z)} {z_orientation[int(z > 0)]}."
                        elif z == 0 and x != 0 and y != 0:
                            order = f"{command} {colour} {abs(x)} {x_orientation[int(x > 0)]} and {abs(y)} {y_orientation[int(y > 0)]}."
                        elif z != 0 and x == 0 and y == 0:
                            order = f"{command} {colour} {abs(z)} {z_orientation[int(z > 0)]}."
                        elif y != 0 and x == 0 and z == 0:
                            order = f"{command} {colour} {abs(y)} {y_orientation[int(y > 0)]}."
                        elif x != 0 and z == 0 and y == 0:
                            order = f"{command} {colour} {abs(x)} {x_orientation[int(x > 0)]}."
                        else:
                            order = f"{command} initial {colour} block."
                        new_history += order
                    else:
                        if command == "put":
                            new_history += f"{command} initial {colour} block."
                            first_block = True
                            x_0, y_0, z_0 = x, y, z
                    prev_line_builder = True
