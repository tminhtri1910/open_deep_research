#!/usr/bin/env python3  
"""Script to get impact radius for GitHub Actions."""  
import json  
import sys  
from pathlib import Path  
  
# Add code-review-graph to path  
sys.path.insert(0, str(Path(__file__).parent.parent))  
  
# pyrefly: ignore [missing-import]
from code_review_graph.tools.query import get_impact_radius  
  
def main():  
    base = sys.argv[1] if len(sys.argv) > 1 else "HEAD~1"  
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 2  
      
    result = get_impact_radius(  
        changed_files=None,  # auto-detect from git  
        max_depth=max_depth,  
        repo_root=None,      # auto-detect  
        base=base,  
        detail_level="standard"  
    )  
      
    print(json.dumps(result, indent=2, default=str))  
  
if __name__ == "__main__":  
    main()