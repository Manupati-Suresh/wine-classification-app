"""
Deployment preparation script for Streamlit Cloud.
Ensures everything is ready for deployment.
"""

import os
import subprocess
import sys

def check_git_repo():
    """Check if this is a git repository."""
    return os.path.exists('.git')

def run_health_check():
    """Run the health check script."""
    try:
        result = subprocess.run([sys.executable, 'health_check.py'], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return True, "Health check passed"
        else:
            return False, f"Health check failed: {result.stderr}"
    except subprocess.TimeoutExpired:
        return False, "Health check timed out"
    except Exception as e:
        return False, str(e)

def check_required_files():
    """Check if all required files exist."""
    required_files = [
        'app.py',
        'requirements.txt',
        'train_model.py',
        'random_forest_model.pkl',
        '.streamlit/config.toml',
        'README.md'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    return len(missing_files) == 0, missing_files

def prepare_deployment():
    """Prepare the app for deployment."""
    print("🚀 Preparing Wine Classification App for Deployment")
    print("=" * 60)
    
    # Check required files
    files_ok, missing = check_required_files()
    if files_ok:
        print("✅ All required files present")
    else:
        print(f"❌ Missing files: {', '.join(missing)}")
        return False
    
    # Run health check
    health_ok, health_output = run_health_check()
    if health_ok:
        print("✅ Health check passed")
    else:
        print("❌ Health check failed")
        print(health_output)
        return False
    
    # Check git repository
    if check_git_repo():
        print("✅ Git repository detected")
        print("📝 Next steps for Streamlit Cloud deployment:")
        print("   1. Commit all changes: git add . && git commit -m 'Deploy wine app'")
        print("   2. Push to GitHub: git push origin main")
        print("   3. Go to https://streamlit.io/cloud")
        print("   4. Connect your GitHub repository")
        print("   5. Deploy with main file: app.py")
    else:
        print("⚠️  No git repository found")
        print("📝 To deploy on Streamlit Cloud:")
        print("   1. Initialize git: git init")
        print("   2. Add files: git add .")
        print("   3. Commit: git commit -m 'Initial commit'")
        print("   4. Create GitHub repository and push")
        print("   5. Deploy on Streamlit Cloud")
    
    print("\n🔗 Deployment URLs:")
    print("   • Streamlit Cloud: https://streamlit.io/cloud")
    print("   • Documentation: https://docs.streamlit.io/streamlit-community-cloud")
    
    print("\n💡 Tips for successful deployment:")
    print("   • Keep requirements.txt minimal and up-to-date")
    print("   • Test locally before deploying: streamlit run app.py")
    print("   • Monitor app logs in Streamlit Cloud dashboard")
    print("   • Use secrets.toml for sensitive configuration")
    
    return True

if __name__ == "__main__":
    success = prepare_deployment()
    if success:
        print("\n🎉 App is ready for deployment!")
    else:
        print("\n❌ Please fix the issues above before deploying.")
        sys.exit(1)