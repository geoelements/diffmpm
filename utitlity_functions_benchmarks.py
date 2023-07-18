import json
import os
import tomllib as tl

def entity_sets(input_file_path: str, output_file_path: str):
    """
        Creates a json file of entity_sets required by diff-mpm
    code from the json file of entity_sets of the CB-Geo MPM code.

    Parameters
    ----------
    input_file_path : str
    output_file_path : str

    Returns
    -------
    None
    """
    if not os.path.exists(input_file_path):
        raise FileNotFoundError("File does not exist")
    f = open(input_file_path)
    data = json.load(f)
    dictionary = {}
    dictionary2 = {}
    for i in data["node_sets"]:
        dictionary2[i["id"]] = i["set"]
    dictionary["node_sets"] = dictionary2
    f.close()
    json_object = json.dumps(dictionary, indent=4)
    with open(output_file_path, "w") as outfile:
        outfile.write(json_object)


def particles_txt_to_json(input_file_path: str, output_file_path: str):
    if not os.path.exists(input_file_path):
        raise FileNotFoundError("File does not exist")
    with open(input_file_path, "r") as f:
        lines = f.readlines()

    data = []
    for line in lines[1:]:
        line = line.strip()
        if line:
            floats = line.split(" ")
        data.append([[float(f) for f in floats]])

    with open(output_file_path, "w") as f:
        json.dump(data, f, indent=2)
