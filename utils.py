import io
import os
import binascii
import numpy as np

label2emotion = {0: "others", 1: "happy", 2: "sad", 3: "angry"}
emotion2label = {"others": 0, "happy": 1, "sad": 2, "angry": 3}


def append_to_file(file_path, content):
    with open(file_path, "a", encoding='utf8') as file:
        file.write(content)
        file.write("\n")


def read_data_file(data_file_path):
    lines = []
    with io.open(data_file_path, encoding="utf8") as finput:
        finput.readline()
        for line in finput:
            line = line.strip().split('\t')
            lines.append(line)
    return lines


def create_solution(predictions, input_Path, suffix='output'):
    output_path = input_Path.replace('.txt', f'_{suffix}.txt')
    with io.open(output_path, "w", encoding="utf8") as fout:
        fout.write('\t'.join(["id", "turn1", "turn2", "turn3", "label"]) + '\n')
        with io.open(input_Path, encoding="utf8") as fin:
            fin.readline()
            for lineNum, line in enumerate(fin):
                fout.write('\t'.join(line.strip().split('\t')[:4]) + '\t')

                prob = predictions[lineNum]
                clazz_idx = np.argmax(prob)

                fout.write(label2emotion[clazz_idx] + '\n')


def create_unique_id():
    return str(binascii.hexlify(os.urandom(16))).replace('b', "").replace("'", "")


def create_directory(checkpoints_dir, unique_id):
    path = checkpoints_dir + '/' + unique_id
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def format_metrics(values):
    formatted_values = []
    for value in values:
        formatted_values.append(str(round(100 * value, 2)))

    return ', '.join(formatted_values)
