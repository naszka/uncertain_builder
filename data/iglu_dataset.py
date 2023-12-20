import pdb
import os
import json
import re
import pandas as pd
import numpy as np
from gridworld.tasks.task import Task, Tasks, Subtasks
from gridworld.data import IGLUDataset, CDMDataset, SingleTurnIGLUDataset
import numpy as np
import pickle
import bz2
import sys
from .utils import (
    color_ambiguity,
    number_ambiguity,
    direction_ambiguity,
    refers_to_past,
    overwrite_referenced_color,
    remove_referenced_color,
    orientation_ambiguity,
    QuestionTask,
)
from collections import defaultdict
from copy import deepcopy

from gridworld.data.load import download
from .resources.heldout_keys import SINGLE_HELDOUT_KEYS

from zipfile import ZipFile
from tqdm import tqdm


VOXELWORLD_GROUND_LEVEL = 63

colors_to_hotbar = {
    "blue": 1,
    "green": 2,
    "red": 3,
    "orange": 4,
    "purple": 5,
    "yellow": 6,
}

hotbar_to_color = {v: k for k, v in colors_to_hotbar.items()}


def fix_xyz(x, y, z):
    XMAX = 11
    YMAX = 9
    ZMAX = 11
    COORD_SHIFT = [5, -63, 5]

    x += COORD_SHIFT[0]
    y += COORD_SHIFT[1]
    z += COORD_SHIFT[2]

    index = z + y * YMAX + x * YMAX * ZMAX
    new_x = index // (YMAX * ZMAX)
    index %= YMAX * ZMAX
    new_y = index // ZMAX
    index %= ZMAX
    new_z = index % ZMAX

    new_x -= COORD_SHIFT[0]
    new_y -= COORD_SHIFT[1]
    new_z -= COORD_SHIFT[2]

    return new_x, new_y, new_z


def fix_log(log_string):
    """
    log_string: str
        log_string should be a string of the full log.
        It should be multiple lines, each corresponded to a timestamp,
        and should be separated by newline character.
    """

    lines = []

    for line in log_string.splitlines():

        if "block_change" in line:
            line_splits = line.split(" ", 2)
            try:
                info = eval(line_splits[2])
            except:
                lines.append(line)
                continue
            x, y, z = info[0], info[1], info[2]
            new_x, new_y, new_z = fix_xyz(x, y, z)
            new_info = (new_x, new_y, new_z, info[3], info[4])
            line_splits[2] = str(new_info)
            fixed_line = " ".join(line_splits)
            # logging.info(f"Fixed {line} to {fixed_line}")

            lines.append(fixed_line)
        else:
            lines.append(line)

    return "\n".join(lines)


class QuestionDataset(SingleTurnIGLUDataset):
    """
    Same datapoints as SingleTurnIGLUDataset plus the datapoints where clarification questions were asked.
    """

    QUESTION_FILENAME = "clarifying_questions_train.csv"
    DATASET_URL = {
        "v0": "https://iglumturkstorage.blob.core.windows.net/public-data/single_turn_dataset.zip",
    }

    def __init__(
        self,
        dataset_version="v0",
        task_kwargs=None,
        force_download=False,
        limit=None,
        parse=False,
        cached_filename=None,
        split=None,
    ) -> None:
        self.split = split
        self.limit = limit
        self.dataset_version = dataset_version
        if dataset_version not in self.DATASET_URL.keys():
            raise Exception(
                "Unknown dataset_version:{} provided.".format(dataset_version)
            )
        if task_kwargs is None:
            task_kwargs = {}
        self.task_kwargs = task_kwargs
        data_path, custom = self.get_data_path()
        if isinstance(self.DATASET_URL[self.dataset_version], tuple):
            filename = self.DATASET_URL[self.dataset_version][1].split("/")[-1]
        else:
            filename = self.DATASET_URL[self.dataset_version].split("/")[-1]
        if custom:
            filename = f"cached_{filename}"
        if cached_filename is not None:
            filename = cached_filename
        if not custom:
            try:
                # first, try downloading the lightweight parsed dataset
                self.download_parsed(
                    data_path=data_path,
                    file_name=filename,
                    force_download=force_download,
                )
                self.load_tasks_dataset(os.path.join(data_path, filename))
            except Exception as e:
                print(e)
                parse = True
        if custom or parse:
            print("Loading parsed dataset failed. Downloading full dataset.")
            # if it fails, download it manually and cache it
            self.download_dataset(data_path, force_download)
            dialogs = self.get_instructions(data_path)
            self.tasks = defaultdict(list)
            self.parse_tasks(dialogs, data_path)
            self.dump_tasks_dataset(os.path.join(data_path, filename))

    def get_clarifying_questions(self):
        path = sys.modules[__name__].__file__
        path_dir, _ = os.path.split(path)
        questions_file = os.path.join(path_dir, "resources", self.QUESTION_FILENAME)
        questions_tabel = pd.read_csv(questions_file)
        return questions_tabel

    def parse_tasks(self, dialogs, path):
        """Fills attribute `self.tasks` with instances of Task.

        A Task contains an initial world state, a target world state and a
        single instruction.

        Parameters
        ----------
        dialogs : pandas.DataFrame
            Contains information of each session, originally stored
            in database tables. The information includes:
                - InitializedWorldStructureId or InitializedWorldGameId:
                  Original target structure id of the initial world.
                - InitializedWorldPath: Path to a json file that contains the
                  initial blocks of the world.
                - ActionDataPath: Path relative to dataset location with the
                  target world.
                - InputInstruction: Session instruction
                - IsHITQualified: boolean indicating if the step is valid.

        path : _type_
            Path with the state of the VoxelWorld grid after each session.
            Each session should have an associated directory named with the
            session id, with json files that describe the world state after
            each step.

        """
        dialogs = dialogs[dialogs.InitializedWorldPath.notna()]
        dialogs["InitializedWorldPath"] = dialogs["InitializedWorldPath"].apply(
            lambda x: x.replace("\\", os.path.sep)
        )
        dialogs["InitializedWorldPath"] = dialogs["InitializedWorldPath"].apply(
            lambda x: x.replace("/", os.path.sep)
        )

        # Get the list of games for which the instructions were clear.

        # Util function to read structure from disk.
        def _load_structure(structure_path):
            filepath = os.path.join(path, structure_path)
            if not os.path.exists(filepath):
                return None

            with open(filepath) as structure_file:
                structure_data = json.load(structure_file)
                blocks = structure_data["worldEndingState"]["blocks"]
                structure = [self.transform_block(block) for block in blocks]

            return structure

        multiturn_dialogs = self.get_multiturn_dialogs(path)
        question_data = self.get_clarifying_questions()

        tasks_count = 0
        pbar = tqdm(
            question_data.iterrows(), total=len(question_data), desc="parsing dataset"
        )
        for _, row in pbar:
            # e.g. initial_world_states\builder-data/8-c92/step-4 -> 8-c92/step-4
            task_id, step_id = row.InitializedWorldPath.split("/")[-2:]
            if self.split == "test":
                if f"{task_id}/{step_id}" not in SINGLE_HELDOUT_KEYS:
                    continue
            if self.split == "train":
                if f"{task_id}/{step_id}" in SINGLE_HELDOUT_KEYS:
                    continue
            # Read initial structure
            initial_world_blocks = _load_structure(row.InitializedWorldPath)
            if initial_world_blocks is None:
                pbar.write(
                    f"Skipping '{row.GameId}'. Can't load starting structure from '{row.InitializedWorldPath}'."
                )
                print(row.IsInstructionClear)
                continue

            found = dialogs[
                (dialogs["GameId"] == row.GameId)
                & (row.InputInstruction == dialogs["InputInstruction"])
            ]
            if len(found) > 0:
                target_world_blocks = _load_structure(found.iloc[0].TargetWorldPath)
                if target_world_blocks is None:
                    pbar.write(
                        f"Skipping '{row.GameId}'. Can't load target structure from '{row.TargetWorldPath}'."
                    )
                    continue
                if sorted(initial_world_blocks) == sorted(target_world_blocks):
                    pbar.write(
                        f"Skipping '{row.GameId}'. Target structure is the same as the initial one."
                    )
                    continue
            else:
                if row.IsInstructionClear == "Yes":
                    # there should be a record of this if it was clear
                    pbar.write(f"Skipping '{row.GameId}'. Target structure not found")
                    continue
                target_world_blocks = []

            last_instruction = "<Architect> " + self.process(row.InputInstruction)
            # Read utterances
            utterances = self.get_previous_dialogs(row, multiturn_dialogs)
            utterances.append(last_instruction)
            utterances = "\n".join(utterances)

            target_label = True if row.IsInstructionClear == "Yes" else False
            # Construct task
            task = self.create_task(
                utterances,
                initial_world_blocks,
                target_world_blocks,
                last_instruction=last_instruction,
                target_label=target_label,
            )

            # self.tasks[row.InitializedWorldStructureId].append(task)
            self.tasks[f"{task_id}/{step_id}"].append(task)
            tasks_count += 1

    def create_task(
        self, previous_chat, initial_grid, target_grid, last_instruction, target_label
    ):
        starting_grid = Tasks.to_sparse(initial_grid)
        task = QuestionTask(
            chat=previous_chat,
            target_grid=Tasks.to_dense(target_grid),
            starting_grid=starting_grid,
            last_instruction=last_instruction,
            is_question=target_label,
        )
        # To properly init max_int and prev_grid_size fields
        task.reset()
        return task


class AmbiguityPairsDataset(QuestionDataset):
    def __init__(
        self,
        dataset_version="v0",
        task_kwargs=None,
        force_download=False,
        limit=None,
        parse=False,
        split=None,
    ) -> None:
        super().__init__(
            dataset_version=dataset_version,
            task_kwargs=task_kwargs,
            force_download=force_download,
            parse=parse,
            split=split,
            limit=limit,
            cached_filename="ambig_single_cached",
        )

    def parse_tasks(self, dialogs, path):
        """Fills attribute `self.tasks` with instances of Task.

        A Task contains an initial world state, a target world state and a
        single instruction.

        Parameters
        ----------
        dialogs : pandas.DataFrame
            Contains information of each session, originally stored
            in database tables. The information includes:
                - InitializedWorldStructureId or InitializedWorldGameId:
                  Original target structure id of the initial world.
                - InitializedWorldPath: Path to a json file that contains the
                  initial blocks of the world.
                - ActionDataPath: Path relative to dataset location with the
                  target world.
                - InputInstruction: Session instruction
                - IsHITQualified: boolean indicating if the step is valid.

        path : _type_
            Path with the state of the VoxelWorld grid after each session.
            Each session should have an associated directory named with the
            session id, with json files that describe the world state after
            each step.

        """
        dialogs = dialogs[dialogs.InitializedWorldPath.notna()]
        dialogs["InitializedWorldPath"] = dialogs["InitializedWorldPath"].apply(
            lambda x: x.replace("\\", os.path.sep)
        )
        dialogs["InitializedWorldPath"] = dialogs["InitializedWorldPath"].apply(
            lambda x: x.replace("/", os.path.sep)
        )

        # Get the list of games for which the instructions were clear.

        # Util function to read structure from disk.
        def _load_structure(structure_path):
            filepath = os.path.join(path, structure_path)
            if not os.path.exists(filepath):
                return None

            with open(filepath) as structure_file:
                structure_data = json.load(structure_file)
                blocks = structure_data["worldEndingState"]["blocks"]
                structure = [self.transform_block(block) for block in blocks]

            return structure

        multiturn_dialogs = self.get_multiturn_dialogs(path)
        question_data = self.get_clarifying_questions()

        row_count = 0
        pbar = tqdm(
            question_data.iterrows(), total=len(question_data), desc="parsing dataset"
        )
        for _, row in pbar:
            # Read initial structure
            initial_world_blocks = _load_structure(row.InitializedWorldPath)
            task_id, step_id = row.InitializedWorldPath.split("/")[-2:]
            if self.split == "test":
                if f"{task_id}/{step_id}" not in SINGLE_HELDOUT_KEYS:
                    continue
            if self.split == "train":
                if f"{task_id}/{step_id}" in SINGLE_HELDOUT_KEYS:
                    continue
            if initial_world_blocks is None:
                pbar.write(
                    f"Skipping '{row.GameId}'. Can't load starting structure from '{row.InitializedWorldPath}'."
                )
                print(row.IsInstructionClear)
                continue

            found = dialogs[
                (dialogs["GameId"] == row.GameId)
                & (row.InputInstruction == dialogs["InputInstruction"])
            ]
            if len(found) > 0:
                target_world_blocks = _load_structure(found.iloc[0].TargetWorldPath)
                if target_world_blocks is None:
                    pbar.write(
                        f"Skipping '{row.GameId}'. Can't load target structure from '{row.TargetWorldPath}'."
                    )
                    continue
                if sorted(initial_world_blocks) == sorted(target_world_blocks):
                    pbar.write(
                        f"Skipping '{row.GameId}'. Target structure is the same as the initial one."
                    )
                    continue
            else:
                target_world_blocks = []

            if row.IsInstructionClear == "No":
                continue
            last_instruction = "<Architect> " + self.process(row.InputInstruction)
            # Read utterances
            utterances = self.get_previous_dialogs(row, multiturn_dialogs)
            utterances.append(last_instruction)
            utterances_ambig = deepcopy(utterances)
            task_orig = self.create_task(
                "\n".join(utterances),
                initial_world_blocks,
                target_world_blocks,
                last_instruction=last_instruction,
                target_label=False,
            )
            self.tasks[f"{task_id}/{step_id}/{row_count}"].append(task_orig)
            # create ambiguity with color
            ambigous = color_ambiguity(last_instruction)
            if len(ambigous.split()) < len(last_instruction.split()):
                utterances_ambig[-1] = ambigous
                task_ambig = self.create_task(
                    "\n".join(utterances_ambig),
                    initial_world_blocks,
                    target_world_blocks,
                    last_instruction=ambigous,
                    target_label=True,
                )
                # e.g. initial_world_states\builder-data/8-c92/step-4 -> 8-c92/step-4
                self.tasks[f"{task_id}/{step_id}/{row_count}_color"].append(task_ambig)

            # create ambiguity with numbers
            ambigous = number_ambiguity(last_instruction)
            if len(ambigous.split()) < len(last_instruction.split()):
                utterances_ambig[-1] = ambigous
                task_ambig = self.create_task(
                    "\n".join(utterances_ambig),
                    initial_world_blocks,
                    target_world_blocks,
                    last_instruction=ambigous,
                    target_label=True,
                )
                # e.g. initial_world_states\builder-data/8-c92/step-4 -> 8-c92/step-4
                self.tasks[f"{task_id}/{step_id}/{row_count}_number"].append(task_ambig)

            # create ambiguity with orientation
            ambigous = orientation_ambiguity(last_instruction)
            if len(ambigous.split()) < len(last_instruction.split()):
                utterances_ambig[-1] = ambigous
                task_ambig = self.create_task(
                    "\n".join(utterances_ambig),
                    initial_world_blocks,
                    target_world_blocks,
                    last_instruction=ambigous,
                    target_label=True,
                )
                # e.g. initial_world_states\builder-data/8-c92/step-4 -> 8-c92/step-4
                self.tasks[f"{task_id}/{step_id}/{row_count}_orientation"].append(
                    task_ambig
                )

            # create ambiguity with direction
            ambigous = direction_ambiguity(last_instruction)
            if len(ambigous.split()) < len(last_instruction.split()):
                utterances_ambig[-1] = ambigous
                task_ambig = self.create_task(
                    "\n".join(utterances_ambig),
                    initial_world_blocks,
                    target_world_blocks,
                    last_instruction=ambigous,
                    target_label=True,
                )
                self.tasks[f"{task_id}/{step_id}/{row_count}_direction"].append(
                    task_ambig
                )

            # create ambiguity with omitting history
            if refers_to_past(last_instruction):
                task_ambig = self.create_task(
                    last_instruction,
                    [],
                    target_world_blocks,
                    last_instruction=last_instruction,
                    target_label=True,
                )
                # e.g. initial_world_states\builder-data/8-c92/step-4 -> 8-c92/step-4
                self.tasks[f"{task_id}/{step_id}/{row_count}_nohistory"].append(
                    task_ambig
                )
                color_overwritten, history_overwritten = overwrite_referenced_color(
                    initial_world_blocks, last_instruction, utterances[:-1]
                )
                history_overwritten.append(last_instruction)
                task_ambig = self.create_task(
                    "\n".join(history_overwritten),
                    color_overwritten,
                    target_world_blocks,
                    last_instruction=last_instruction,
                    target_label=True,
                )
                # e.g. initial_world_states\builder-data/8-c92/step-4 -> 8-c92/step-4
                self.tasks[
                    f"{task_id}/{step_id}/{row_count}_historycolorchanged"
                ].append(task_ambig)
                blocks_overwritten, history_overwritten = remove_referenced_color(
                    initial_world_blocks, last_instruction, utterances[:-1]
                )
                history_overwritten.append(last_instruction)
                task_ambig = self.create_task(
                    "\n".join(history_overwritten),
                    blocks_overwritten,
                    target_world_blocks,
                    last_instruction=last_instruction,
                    target_label=True,
                )
                # e.g. initial_world_states\builder-data/8-c92/step-4 -> 8-c92/step-4
                self.tasks[
                    f"{task_id}/{step_id}/{row_count}_historycolorremoved"
                ].append(task_ambig)

            row_count += 1


class AmbiguityPairsDatasetMulti(IGLUDataset):
    def __init__(
        self,
        dataset_version="v0.1.0-rc3",
        task_kwargs=None,
        force_download=False,
        limit=None,
        parse=False,
    ) -> None:
        self.limit = limit
        super().__init__(
            dataset_version=dataset_version,
            task_kwargs=task_kwargs,
            force_download=force_download,
            parse=parse,
            cached_filename="ambig_multi_cached",
        )

    def create_task(
        self, previous_chat, initial_grid, target_grid, last_instruction, target_label
    ):
        starting_grid = Tasks.to_sparse(initial_grid)
        task = QuestionTask(
            chat=previous_chat,
            target_grid=Tasks.to_dense(target_grid),
            starting_grid=starting_grid,
            last_instruction=last_instruction,
        )
        # To properly init max_int and prev_grid_size fields
        task.reset()
        return task

    def parse_tasks(self, dialogs, path):
        """Fills attribute `self.tasks` with utterances from `dialogs` and
        VoxelWorld states for each step.

        Parameters
        ----------
        dialogs : pandas.DataFrame
            Contains information of each turn in the session, originally stored
            in database tables. The information includes:
                - PartitionKey: corresponds to Game attempt or session. It is
                  constructed following the pattern `{attemptId}-{taskId}`
                - structureId: task id of the session.
                - StepId: number of step in the session. For multi-turn IGLU
                  data, all odd steps have type architect and even steps
                  have type builder. Depending on the task type, different
                  columns will be used to fill the task.
                - IsHITQualified: boolean indicating if the step is valid.

        path : str
            Path with the state of the VoxelWorld grid after each session.
            Each session should have an associated directory named with the
            session id, with json files that describe the world state after
            each step.

        """
        # Partition key
        groups = dialogs.groupby("PartitionKey")
        task_count = 0
        for sess_id, gr in tqdm(groups, total=len(groups), desc="parsing dataset"):
            # This corresponds to the entire dialog between steps with
            # changes to the blocks
            utt_seq = []
            blocks = []
            if not os.path.exists(f"{path}/builder-data/{sess_id}"):
                continue
            # Each session should have a single taskId associated.
            assert len(gr.structureId.unique()) == 1
            structure_id = gr.structureId.values[0]
            # Read the utterances and block end positions for each step.
            for i, row in gr.sort_values("StepId").reset_index(drop=True).iterrows():
                if not row.IsHITQualified:
                    continue
                if row.StepId % 2 == 1:
                    # Architect step
                    if isinstance(row.instruction, str):
                        utt_seq.append([])
                        utt_seq[-1].append(
                            f"<Architect> {self.process(row.instruction)}"
                        )
                    elif isinstance(row.Answer4ClarifyingQuestion, str):
                        utt_seq[-1].append(
                            f"<Architect> {self.process(row.Answer4ClarifyingQuestion)}"
                        )
                else:
                    # Builder step
                    if isinstance(row.ClarifyingQuestion, str):
                        utt_seq[-1].append(
                            f"<Builder> {self.process(row.ClarifyingQuestion)}"
                        )
                        continue
                    blocks.append([])
                    curr_step = f"{path}/builder-data/{sess_id}/step-{row.StepId}"
                    if not os.path.exists(curr_step):
                        break
                        # TODO: in this case the multiturn collection was likely
                        # "reset" so we need to stop parsing this session. Need to check that.
                    with open(curr_step) as f:
                        step_data = json.load(f)
                    for block in step_data["worldEndingState"]["blocks"]:
                        x, y, z, bid = self.transform_block(block)
                        blocks[-1].append((x, y, z, bid))
            # Aggregate all previous blocks into each step
            if len(blocks) < len(utt_seq):
                # handle the case of missing of the last blocks record
                utt_seq = utt_seq[: len(blocks)]
            i = 0
            while i < len(blocks):
                # Collapse steps where there are no block changes.
                if len(blocks[i]) == 0:
                    if i == len(blocks) - 1:
                        blocks = blocks[:i]
                        utt_seq = utt_seq[:i]
                    else:
                        blocks = blocks[:i] + blocks[i + 1 :]
                        utt_seq[i] = utt_seq[i] + utt_seq[i + 1]
                        utt_seq = utt_seq[: i + 1] + utt_seq[i + 2 :]
                else:
                    i += 1
            if len(blocks) > 0:
                # Create random subtasks from the sequence of dialogs and blocks
                subtasks = Subtasks(utt_seq, blocks, **self.task_kwargs)
                assert len(utt_seq) == len(blocks)
                subtasks.full_grid = np.full((9, 11, 11), 1)
                for sub_task_id in range(-1, len(subtasks.dialog) - 1):
                    task = subtasks.create_task(sub_task_id, sub_task_id + 1)
                    task_count += 1
                    self.tasks[f"{structure_id}/{task_count}"].append(task)

                    chat_history = task.chat.replace(task.last_instruction, "").strip(
                        "\n"
                    )
                    last_instruction = task.last_instruction
                    # create ambiguity with color
                    ambigous = color_ambiguity(last_instruction)
                    if len(ambigous.split()) < len(last_instruction.split()):
                        task_ambig = self.create_task(
                            "\n".join([chat_history, ambigous]),
                            task.starting_grid,
                            task.target_grid,
                            last_instruction=ambigous,
                            target_label=0,
                        )
                        self.tasks[f"{structure_id}/{task_count}_color"].append(
                            task_ambig
                        )

                    # create ambiguity with numbers
                    ambigous = number_ambiguity(last_instruction)
                    if len(ambigous.split()) < len(last_instruction.split()):
                        task_ambig = self.create_task(
                            "\n".join([chat_history, ambigous]),
                            task.starting_grid,
                            task.target_grid,
                            last_instruction=ambigous,
                            target_label=0,
                        )
                        self.tasks[f"{structure_id}/{task_count}_number"].append(
                            task_ambig
                        )

                    # create ambiguity with orientation
                    ambigous = orientation_ambiguity(last_instruction)
                    if len(ambigous.split()) < len(last_instruction.split()):
                        task_ambig = self.create_task(
                            "\n".join([chat_history, ambigous]),
                            task.starting_grid,
                            task.target_grid,
                            last_instruction=ambigous,
                            target_label=0,
                        )
                        self.tasks[f"{structure_id}/{task_count}_orientation"].append(
                            task_ambig
                        )

                    # create ambiguity with emitting history
                    if refers_to_past(last_instruction):
                        task_ambig = self.create_task(
                            last_instruction,
                            [],
                            task.target_grid,
                            last_instruction=last_instruction,
                            target_label=0,
                        )
                        # e.g. initial_world_states\builder-data/8-c92/step-4 -> 8-c92/step-4
                        self.tasks[f"{structure_id}/{task_count}_nohistory"].append(
                            task_ambig
                        )

                        (
                            color_overwritten,
                            history_overwritten,
                        ) = overwrite_referenced_color(
                            task.starting_grid,
                            last_instruction,
                            chat_history.split("\n"),
                        )
                        history_overwritten.append(last_instruction)
                        task_ambig = self.create_task(
                            "\n".join(history_overwritten),
                            color_overwritten,
                            task.target_grid,
                            last_instruction=last_instruction,
                            target_label=0,
                        )
                        # e.g. initial_world_states\builder-data/8-c92/step-4 -> 8-c92/step-4
                        self.tasks[
                            f"{structure_id}/{task_count}_historycolorchanged"
                        ].append(task_ambig)

                        (
                            blocks_overwritten,
                            history_overwritten,
                        ) = remove_referenced_color(
                            task.starting_grid,
                            last_instruction,
                            chat_history.split("\n"),
                        )
                        history_overwritten.append(last_instruction)
                        task_ambig = self.create_task(
                            "\n".join(history_overwritten),
                            blocks_overwritten,
                            task.target_grid,
                            last_instruction=last_instruction,
                            target_label=0,
                        )
                        # e.g. initial_world_states\builder-data/8-c92/step-4 -> 8-c92/step-4
                        self.tasks[
                            f"{structure_id}/{task_count}_historycolorremoved"
                        ].append(task_ambig)
