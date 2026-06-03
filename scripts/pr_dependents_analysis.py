#!/usr/bin/env python3  
"""Script to analyze PR changes and dependents for GitHub Actions."""  
import json  
import sys  
from pathlib import Path  
  
sys.path.insert(0, str(Path(__file__).parent.parent))  
  
from code_review_graph.changes import analyze_changes, parse_diff_ranges  
from code_review_graph.tools.query import get_impact_radius  
from code_review_graph.graph import GraphStore  
from code_review_graph.incremental import get_changed_files  
  
def main():  
    repo_root = Path.cwd()  
    base = sys.argv[1] if len(sys.argv) > 1 else "HEAD~1"  
      
    # Open graph store  
    store = GraphStore(repo_root / ".code-review-graph" / "graph.db")  
      
    try:  
        # Get changed files  
        changed_files = get_changed_files(repo_root, base)  
        if not changed_files:  
            print(json.dumps({"status": "ok", "summary": "No changes detected."}))  
            return  
          
        # # Parse diff ranges for line-level precision  
        # diff_ranges = parse_diff_ranges(str(repo_root), base)  
        # abs_ranges = {}  
        # for rel_path, ranges in diff_ranges.items():  
        #     abs_path = str(repo_root / rel_path)  
        #     abs_ranges[abs_path] = ranges  
          
        # Analyze changes with line-level precision  
        analysis = analyze_changes(  
            store,  
            changed_files=changed_files,  
            # changed_ranges=abs_ranges if abs_ranges else None,  
            changed_ranges=None,  
            repo_root=str(repo_root),  
            base=base,  
        )  
          
        # Get impact radius for dependents  
        impact = get_impact_radius(  
            changed_files=changed_files,  
            max_depth=1,  
            repo_root=str(repo_root),  
            base=base,  
            detail_level="standard"  
        )  
          
        # Combine results  
        result = {  
            "status": "ok",  
            "summary": analysis["summary"],  
            "risk_score": analysis["risk_score"],  
            "changed_functions": analysis["changed_functions"],  
            "affected_flows": analysis["affected_flows"],  
            "test_gaps": analysis["test_gaps"],  
            "review_priorities": analysis["review_priorities"],  
            "impacted_nodes": impact["impacted_nodes"],  
            "impacted_files": impact["impacted_files"],  
            "impacted_file_count": len(impact["impacted_files"]),  
        }  
          
        print(json.dumps(result, indent=2, default=str))  
    finally:  
        store.close()  
  
if __name__ == "__main__":  
    main()