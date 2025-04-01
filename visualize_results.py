import matplotlib.pyplot as plt
import pandas as pd
import json
import numpy as np
from typing import List, Dict, Any
import seaborn as sns


def visualize_westworld_qa_results(log_file: str):
    """
    Create visualizations from Westworld QA test results for refund policy compliance.
    
    Args:
        log_file: Path to the JSONL file with test results
    """
    # Load the results
    results = []
    with open(log_file, 'r') as f:
        for line in f:
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    if not results:
        print("No results found in log file.")
        return
    
    # Create a DataFrame for easier analysis
    df = pd.DataFrame([
        {
            'category': r['category'],
            'policy_violation': r['evaluation'].get('policy_violation', False),
            'severity': r['evaluation'].get('severity', 0),
            'violation_type': r['evaluation'].get('violation_category', 'none'),
            'timestamp': r['timestamp']
        }
        for r in results
    ])
    
    # Set up the visualization style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("muted")
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(15, 12))
    fig.suptitle('Retail Refund Policy Compliance Analysis', fontsize=16, fontweight='bold')
    
    # 1. Violation Rates by Attack Category
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2)
    category_results = df.groupby('category')['policy_violation'].agg(['count', 'sum'])
    category_results['rate'] = category_results['sum'] / category_results['count']
    category_results = category_results.sort_values('rate', ascending=False)
    
    sns.barplot(
        x=category_results.index,
        y=category_results['rate'],
        ax=ax1
    )
    ax1.set_title('Policy Violation Rate by Customer Request Type')
    ax1.set_xlabel('Request Type')
    ax1.set_ylabel('Violation Rate')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # 2. Violation Types Distribution
    ax2 = plt.subplot2grid((3, 3), (0, 2))
    violations_df = df[df['policy_violation'] == True]
    if not violations_df.empty and 'violation_type' in violations_df.columns:
        violation_counts = violations_df['violation_type'].value_counts()
        violation_counts = violation_counts[violation_counts.index != 'none']
        if not violation_counts.empty:
            violation_counts.plot(kind='pie', ax=ax2, autopct='%1.1f%%')
            ax2.set_title('Types of Policy Violations')
            ax2.set_ylabel('')
        else:
            ax2.text(0.5, 0.5, 'No specific violation types found', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax2.transAxes)
    else:
        ax2.text(0.5, 0.5, 'No violations detected', 
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax2.transAxes)
    
    # 3. Severity Distribution
    ax3 = plt.subplot2grid((3, 3), (1, 0))
    if not violations_df.empty:
        sns.histplot(violations_df['severity'], bins=10, kde=True, ax=ax3)
        ax3.set_title('Violation Severity Distribution')
        ax3.set_xlabel('Severity Score')
        ax3.set_ylabel('Count')
    else:
        ax3.text(0.5, 0.5, 'No violations detected', 
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax3.transAxes)
    
    # 4. Success Rate by Request Type (horizontal bar)
    ax4 = plt.subplot2grid((3, 3), (1, 1), colspan=2)
    if not category_results.empty:
        # Sort by success rate (inverse of violation rate)
        category_results['success_rate'] = 1 - category_results['rate']
        category_results = category_results.sort_values('success_rate')
        
        sns.barplot(
            y=category_results.index,
            x=category_results['success_rate'],
            ax=ax4,
            palette="YlGnBu"
        )
        ax4.set_title('Policy Compliance Rate by Request Type')
        ax4.set_ylabel('Request Type')
        ax4.set_xlabel('Compliance Rate')
        ax4.set_xlim(0, 1)
        
        # Add percentage labels
        for i, v in enumerate(category_results['success_rate']):
            ax4.text(v + 0.01, i, f'{v:.1%}', va='center')
    else:
        ax4.text(0.5, 0.5, 'No category data available', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax4.transAxes)
    
    # 5. Overall Summary Statistics
    ax5 = plt.subplot2grid((3, 3), (2, 0), colspan=3)
    overall_violation_rate = df['policy_violation'].mean()
    avg_severity = violations_df['severity'].mean() if not violations_df.empty else df.empty
    
    # Create a text summary
    summary_text = (
        f"OVERALL RESULTS:\n\n"
        f"Total Tests: {len(df)}\n"
        f"Policy Violations: {len(violations_df)} ({overall_violation_rate:.1%})\n"
        f"Average Violation Severity: {avg_severity:.2f}/10\n\n"
    )
    
    # Add information about most common violation types
    if not violations_df.empty and 'violation_type' in violations_df.columns:
        top_violations = violations_df['violation_type'].value_counts().head(3)
        if not top_violations.empty and top_violations.index[0] != 'none':
            summary_text += "Most Common Violation Types:\n"
            for vtype, count in top_violations.items():
                if vtype != 'none':
                    summary_text += f"- {vtype}: {count} occurrences ({count/len(violations_df):.1%} of violations)\n"
    
    ax5.axis('off')
    ax5.text(0.5, 0.5, summary_text, 
             horizontalalignment='center', verticalalignment='center',
             transform=ax5.transAxes, fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=1'))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('westworld_qa_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Visualization saved as 'westworld_qa_results.png'")
    
    # Create a summary of interesting examples
    create_interesting_examples_summary(results)


def create_interesting_examples_summary(results: List[Dict[str, Any]]):
    """Create a summary of the most interesting test cases."""
    # Filter for violations only
    violations = [r for r in results if r['evaluation'].get('policy_violation', False)]
    
    if not violations:
        print("No violations detected in the test results.")
        return
    
    # Sort by severity
    violations.sort(key=lambda x: x['evaluation'].get('severity', 0), reverse=True)
    
    # Get the top 3 most severe violations
    top_violations = violations[:min(3, len(violations))]
    
    with open('interesting_examples.md', 'w') as f:
        f.write("# Most Interesting Test Cases\n\n")
        
        for i, violation in enumerate(top_violations, 1):
            f.write(f"## Example {i}: {violation['category']}\n\n")
            f.write(f"**Severity:** {violation['evaluation'].get('severity', 0)}/10\n\n")
            
            # For multi-turn conversations
            if 'full_conversation' in violation:
                f.write("**Full Conversation:**\n```\n{}\n```\n\n".format(violation['full_conversation']))
            else:
                # For single-turn conversations
                if 'attack_prompt' in violation:
                    f.write("**Customer Request:**\n```\n{}\n```\n\n".format(violation['attack_prompt']))
                elif 'initial_attack' in violation:
                    f.write("**Initial Customer Request:**\n```\n{}\n```\n\n".format(violation['initial_attack']))
                
                if 'target_response' in violation:
                    f.write("**Chatbot Response:**\n```\n{}\n```\n\n".format(violation['target_response']))
                elif 'final_response' in violation:
                    f.write("**Final Chatbot Response:**\n```\n{}\n```\n\n".format(violation['final_response']))
            
            f.write("**Violation Category:** {}\n\n".format(violation['evaluation'].get('violation_category', 'Unknown')))
            f.write("**Evaluation:**\n{}\n\n".format(violation['evaluation'].get('explanation', 'No explanation provided')))
            
            improvement_key = next((k for k in ['improvement', 'recommendation'] if k in violation['evaluation']), None)
            if improvement_key:
                f.write("**Recommendation:**\n{}\n\n".format(violation['evaluation'][improvement_key]))
            
            f.write("---\n\n")
    
    print("Interesting examples summary saved as 'interesting_examples.md'")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize Westworld QA test results')
    parser.add_argument('log_file', type=str, help='Path to the JSONL log file')
    args = parser.parse_args()
    
    visualize_westworld_qa_results(args.log_file)
