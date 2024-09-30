import csv
import re

def extract_info(file_path, output_path):
    with open(file_path, 'r') as file, open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['CurrentIndex', 'HSJA_Logits_True/False', 'SortedList_First_Element', 'SortedList_Second_Element'])
        
        current_index = None
        
        for line in file:
            line = line.strip()
            
            # Check if the line contains "currentIndex="
            if "currentIndex=" in line:
                match = re.search(r'currentIndex= (\d+)', line)
                if match:
                    current_index = match.group(1)
                    
            # Check if the line starts with "HSJA logits"
            elif line.startswith("HSJA logits"):
                match = re.search(r'HSJA logits, is_cache_i= (True|False)', line)
                if match:
                    hsja_value = match.group(1)
            
            # Check if the line starts with "sortedList="
            elif line.startswith("sortedList="):
                match = re.search(r'sortedList= \[([0-9e.-]+), ([0-9e.-]+), ', line)
                if match:
                    first_element = match.group(1)
                    second_element = match.group(2)
                    writer.writerow([current_index if current_index else '', hsja_value, first_element, second_element])
                    current_index = None

if __name__ == "__main__":
    file_path = 'hsja-noop.out'
    output_path = 'hsja-noop-data.csv'
    extract_info(file_path, output_path)
