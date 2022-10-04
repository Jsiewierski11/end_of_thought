import json
import random
import csv

def read_json(filepath):
    f = open(filepath)
    dialogues = json.loads(f.read())
    return dialogues

def convert_to_list(string):
    li = string.split(" ")
    return li

def get_random_sample(string):
    li = string.split(" ")

    num = random.randrange(0, len(li))
    if num == 0: num += 1
    new_lst = li[:num]
    new_str = " ".join(new_lst)
    return new_str

def generate_data(writer, dialogues):
    dataset = []
    for dialog in dialogues:
        for utternce in dialog['dialog']:
            dataset.append((utternce[1], 1))
            writer.writerow((utternce[1], "1"))
            print((utternce[1], 1))

            random_sample = get_random_sample(utternce[1])
            dataset.append((random_sample, 0))
            writer.writerow((random_sample, "0"))
            print((random_sample, 0))
    return dataset

if __name__ == "__main__":
    test_dialogues = read_json('./blended_skill_talk/test.json')

    print(test_dialogues[0]['dialog'][0][1])
    with open('./data/test.tsv', 'w') as f:
        writer = csv.writer(f, delimiter="\t")
        dataset = generate_data(writer, test_dialogues)
    print(len(dataset))

    train_dialogues = read_json('./blended_skill_talk/train.json')
    with open('./data/train.tsv', 'w') as f:
        writer = csv.writer(f, delimiter="\t")
        dataset = generate_data(writer, train_dialogues)
    print(len(dataset))

    valid_dialogues = read_json('./blended_skill_talk/valid.json')
    with open('./data/valid.tsv', 'w') as f:
        writer = csv.writer(f, delimiter="\t")
        dataset = generate_data(writer, valid_dialogues)
    print(len(dataset))