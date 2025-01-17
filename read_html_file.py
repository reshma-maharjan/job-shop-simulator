import re
from bs4 import BeautifulSoup
from collections import defaultdict

def analyze_job_shop_schedule(html_file_path):
    """
    Analyzes a job shop schedule HTML file to verify completeness and extract statistics.
    
    Args:
        html_file_path (str): Path to the HTML file containing the job shop schedule
        
    Returns:
        dict: Dictionary containing analysis results
    """
    # Read HTML file
    with open(html_file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Extract machine information
    machine_rows = soup.find_all('div', class_='machine-row')
    num_machines = len(machine_rows)
    
    # Extract machine utilization
    machine_utils = {}
    for row in machine_rows:
        machine_label = row.find('div', class_='machine-label').text
        machine_num = re.search(r'Machine (\d+)', machine_label).group(1)
        util = re.search(r'\(([\d.]+)% util\)', machine_label).group(1)
        machine_utils[f"Machine {machine_num}"] = float(util)
    
    # Extract operations
    operations = re.findall(r'J(\d+)-(\d+)', html_content)
    
    # Analyze jobs and their operations
    job_operations = defaultdict(set)
    for job, operation in operations:
        job_operations[f"Job {job}"].add(int(operation))
    
    # Check completeness of operations for each job
    jobs_complete = {}
    for job, ops in job_operations.items():
        ops_list = sorted(list(ops))
        expected_ops = list(range(max(ops_list) + 1))
        jobs_complete[job] = (ops_list == expected_ops)
    
    # Extract critical path operations
    critical_path_ops = re.findall(r'critical-path.*?J(\d+)-(\d+)', html_content)
    
    # Prepare analysis results
    results = {
        'num_machines': num_machines,
        'num_jobs': len(job_operations),
        'total_operations': len(operations),
        'unique_operations': len(set(operations)),
        'machine_utilization': machine_utils,
        'jobs_operations': {job: sorted(list(ops)) for job, ops in job_operations.items()},
        'jobs_complete': jobs_complete,
        'critical_path': [(int(job), int(op)) for job, op in critical_path_ops],
        'schedule_complete': all(jobs_complete.values())
    }
    
    return results

def print_analysis(results):
    """
    Prints the analysis results in a readable format.
    
    Args:
        results (dict): Analysis results from analyze_job_shop_schedule
    """
    print("Job Shop Schedule Analysis")
    print("=" * 50)
    print(f"\nGeneral Statistics:")
    print(f"Number of machines: {results['num_machines']}")
    print(f"Number of jobs: {results['num_jobs']}")
    print(f"Total operations scheduled: {results['total_operations']}")
    print(f"Unique operations: {results['unique_operations']}")
    
    print("\nMachine Utilization:")
    for machine, util in sorted(results['machine_utilization'].items()):
        print(f"{machine}: {util}%")
    
    print("\nJobs and Operations:")
    for job, operations in sorted(results['jobs_operations'].items()):
        complete = "✓" if results['jobs_complete'][job] else "✗"
        print(f"{job} {complete}: Operations {operations}")
    
    print("\nCritical Path:")
    for job, op in results['critical_path']:
        print(f"Job {job} Operation {op}")
    
    print(f"\nSchedule Completeness: {'Complete' if results['schedule_complete'] else 'Incomplete'}")

if __name__ == "__main__":
    # Example usage
    #html_file_path = "output/doris_ql_gantt.html"
    html_file_path = "ta70_qlearning_gantt.html"
    try:
        results = analyze_job_shop_schedule(html_file_path)
        print_analysis(results)
    except FileNotFoundError:
        print(f"Error: Could not find file {html_file_path}")
    except Exception as e:
        print(f"Error analyzing schedule: {str(e)}")