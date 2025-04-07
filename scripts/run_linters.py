#!/usr/bin/env python3
"""
WDBX Linting Runner

This script runs both the general linting tool and the Discord bot-specific fixer,
then generates a report on what was fixed and what might need manual attention.
"""

import os
import sys
import subprocess
import logging
import argparse
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run WDBX linters and generate a report")
    parser.add_argument('--target', '-t', default='.', help='Target directory or file to process')
    parser.add_argument('--fix', '-f', action='store_true', help='Apply fixes (without this flag, only reports issues)')
    parser.add_argument('--report', '-r', action='store_true', help='Generate HTML report')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show more detailed output')
    return parser.parse_args()

def run_general_linter(target, fix=False, verbose=False):
    """Run the general WDBX linting tool."""
    logger.info("Running general WDBX linter...")
    
    cmd = [sys.executable, 'scripts/wdbx_linter.py', '--target', target]
    if fix:
        cmd.append('--fix')
    if verbose:
        cmd.append('--verbose')
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            logger.warning("General linter encountered some issues:")
            if verbose:
                logger.info(result.stdout)
                logger.error(result.stderr)
            else:
                # Just show a summary of the issues
                for line in result.stdout.splitlines():
                    if line.startswith(('INFO:', 'WARNING:', 'ERROR:')):
                        logger.info(line)
        else:
            logger.info("General linter ran successfully")
            if verbose:
                logger.info(result.stdout)
        
        return {
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
    except Exception as e:
        logger.error(f"Error running general linter: {e}")
        return {
            'success': False,
            'stdout': '',
            'stderr': str(e)
        }

def run_discord_bot_fixer(fix=False, verbose=False):
    """Run the Discord bot-specific fixer."""
    logger.info("Running Discord bot fixer...")
    
    discord_fixer_path = 'scripts/fix_discord_bot.py'
    if not os.path.exists(discord_fixer_path):
        logger.warning(f"Discord bot fixer not found at {discord_fixer_path}")
        return {
            'success': False,
            'stdout': '',
            'stderr': f"Discord bot fixer not found at {discord_fixer_path}"
        }
    
    try:
        # Run in dry-run mode if fix is False
        if not fix:
            # The fix_discord_bot.py script doesn't have a --fix flag, so we need to modify it
            with open(discord_fixer_path, 'r') as f:
                content = f.read()
            
            # Find the fix_discord_bot function and add a dry run check
            if 'def fix_discord_bot():' in content and 'with open(discord_bot_path, \'w\'' in content:
                # Create a temporary version with dry run option
                temp_path = 'scripts/fix_discord_bot_temp.py'
                modified_content = content.replace(
                    'def fix_discord_bot():',
                    'def fix_discord_bot(dry_run=False):'
                ).replace(
                    'if content != original_content:',
                    'if content != original_content and not dry_run:'
                ).replace(
                    'success = fix_discord_bot()',
                    'import sys\ndry_run = "--dry-run" in sys.argv\nsuccess = fix_discord_bot(dry_run)'
                )
                
                with open(temp_path, 'w') as f:
                    f.write(modified_content)
                
                cmd = [sys.executable, temp_path]
                if not fix:
                    cmd.append('--dry-run')
                
                result = subprocess.run(cmd, capture_output=True, text=True, check=False)
                
                # Clean up temporary file
                os.remove(temp_path)
            else:
                # If we can't safely modify the script, run it only if fix is True
                if fix:
                    result = subprocess.run([sys.executable, discord_fixer_path], 
                                          capture_output=True, text=True, check=False)
                else:
                    logger.warning("Cannot run Discord bot fixer in dry-run mode, skipping")
                    return {
                        'success': False,
                        'stdout': '',
                        'stderr': "Cannot run in dry-run mode, needs --fix to apply changes"
                    }
        else:
            # Run normally if fix is True
            result = subprocess.run([sys.executable, discord_fixer_path], 
                                  capture_output=True, text=True, check=False)
        
        if result.returncode != 0:
            logger.warning("Discord bot fixer encountered some issues:")
            if verbose:
                logger.info(result.stdout)
                logger.error(result.stderr)
            else:
                # Just show a summary
                for line in result.stdout.splitlines():
                    if ":" in line:
                        logger.info(line)
        else:
            logger.info("Discord bot fixer ran successfully")
            if verbose:
                logger.info(result.stdout)
        
        return {
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
    
    except Exception as e:
        logger.error(f"Error running Discord bot fixer: {e}")
        return {
            'success': False,
            'stdout': '',
            'stderr': str(e)
        }

def generate_report(general_result, discord_result, target, fix_applied):
    """Generate an HTML report of linting results."""
    report_dir = Path('linting_reports')
    report_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = report_dir / f"linting_report_{timestamp}.html"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>WDBX Linting Report - {timestamp}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #333; }}
            .section {{ margin-bottom: 30px; }}
            .success {{ color: green; }}
            .warning {{ color: orange; }}
            .error {{ color: red; }}
            pre {{ background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
        </style>
    </head>
    <body>
        <h1>WDBX Linting Report</h1>
        <div class="section">
            <h2>Summary</h2>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Target:</strong> {target}</p>
            <p><strong>Fix Applied:</strong> {'Yes' if fix_applied else 'No (Dry Run)'}</p>
            <p><strong>General Linter Status:</strong> 
                <span class="{'success' if general_result['success'] else 'error'}">
                    {'Success' if general_result['success'] else 'Failed'}
                </span>
            </p>
            <p><strong>Discord Bot Fixer Status:</strong> 
                <span class="{'success' if discord_result['success'] else 'error'}">
                    {'Success' if discord_result['success'] else 'Failed'}
                </span>
            </p>
        </div>
        
        <div class="section">
            <h2>General Linter Output</h2>
            <pre>{general_result['stdout']}</pre>
            {f'<h3 class="error">Errors</h3><pre>{general_result["stderr"]}</pre>' if general_result['stderr'] else ''}
        </div>
        
        <div class="section">
            <h2>Discord Bot Fixer Output</h2>
            <pre>{discord_result['stdout']}</pre>
            {f'<h3 class="error">Errors</h3><pre>{discord_result["stderr"]}</pre>' if discord_result['stderr'] else ''}
        </div>
        
        <div class="section">
            <h2>Recommendations</h2>
            <ul>
                {'<li>Apply fixes with <code>--fix</code> flag</li>' if not fix_applied else ''}
                <li>Review any remaining issues manually</li>
                <li>Run tests to ensure fixes didn't break functionality</li>
                <li>Consider using IDE-specific linting tools (e.g., pylint, flake8) for more comprehensive checks</li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"Linting report generated at {report_path}")
    return report_path

def main():
    """Main entry point for the script."""
    args = parse_arguments()
    
    # Run general linter
    general_result = run_general_linter(args.target, args.fix, args.verbose)
    
    # Run Discord bot-specific fixer
    discord_result = run_discord_bot_fixer(args.fix, args.verbose)
    
    # Generate report if requested
    if args.report:
        report_path = generate_report(general_result, discord_result, args.target, args.fix)
        print(f"\nReport generated at: {report_path}")
    
    # Return success only if both tools succeeded
    return 0 if general_result['success'] and discord_result['success'] else 1

if __name__ == "__main__":
    sys.exit(main()) 