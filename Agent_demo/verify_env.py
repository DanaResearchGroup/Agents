try:
    import pydantic_ai
    print("✅ pydantic-ai is installed.")
except ImportError:
    print("❌ pydantic-ai is NOT installed.")

try:
    import rdkit
    from rdkit import Chem
    print("✅ rdkit is installed.")
except ImportError:
    print("❌ rdkit is NOT installed.")

try:
    import dotenv
    print("✅ python-dotenv is installed.")
except ImportError:
    print("❌ python-dotenv is NOT installed.")

try:
    import google.generativeai
    print("✅ google-generativeai is installed.")
except ImportError:
    print("❌ google-generativeai is NOT installed.")

try:
    import openai
    print("✅ openai is installed.")
except ImportError:
    print("❌ openai is NOT installed.")

print("\nIf all checks passed, you are ready to run main.py!")
