
def get_current_directory():
    from pathlib import Path
    current_directory = Path(__file__).resolve().parent
    return current_directory


def get_taillard_instance(instance_name):
    from pathlib import Path
    current_directory = get_current_directory()
    instance_path = current_directory / "taillard_instances" / instance_name + ".txt"
    return instance_path

def get_problem_instance(instance_name):
    from pathlib import Path
    current_directory = get_current_directory()
    instance_path = current_directory / "problem_instances" / instance_name
    return instance_path