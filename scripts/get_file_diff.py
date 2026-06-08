import subprocess  
import json  
import sys

base_ref = sys.argv[1] if len(sys.argv) > 1 else "HEAD~1"

# Chạy git diff với -U0 (giống detect_changes)  
diff_output = subprocess.run(  
    ['git', 'diff', base_ref, '-U0'],  
    capture_output=True,  
    text=True  
).stdout  
    
# Parse diff hunks thành FileDiff structure  
files = []  
current = None  
    
for line in diff_output.split('\n'):  
    if line.startswith('+++ b/'):  
        current = {'filePath': line[6:], 'hunks': []}  
        files.append(current)  
    elif line.startswith('@@') and current:  
        import re  
        match = re.match(r'@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@', line)  
        if match:  
            start = int(match.group(1))  
            count = int(match.group(2)) if match.group(2) else 1  
            if count > 0:       
                current['hunks'].append({  
                    'startLine': start,  
                    'endLine': start + count - 1  
                })  
    
# Lưu kết quả  
with open('file-diffs.json', 'w') as f:  
    json.dump(files, f, indent=2)  
    
print(f"Parsed {len(files)} files")  

with open('file-diffs.json') as f:  
        files = json.load(f)  
      
for file_diff in files:  
    print(f"\nFile: {file_diff['filePath']}")  
    for hunk in file_diff['hunks']:  
        print(f"  Hunk: lines {hunk['startLine']}-{hunk['endLine']}")