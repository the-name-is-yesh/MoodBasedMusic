import os
import pandas as pd


CSV_PATH = "data/ckextended.csv"


print("========================================")
print("        CK+ DATASET INSPECTION")
print("========================================")


# ----------------------------------------
# 1. Check whether the file exists
# ----------------------------------------

if not os.path.exists(CSV_PATH):
    print("\nERROR: CK+ CSV file was not found.")
    print("Expected location:")
    print(CSV_PATH)
    raise SystemExit


print("\nCSV file found!")
print("Location:", CSV_PATH)


# ----------------------------------------
# 2. Load the CSV
# ----------------------------------------

print("\nLoading CK+ dataset...")

data = pd.read_csv(CSV_PATH)

print("Dataset loaded successfully!")


# ----------------------------------------
# 3. Basic information
# ----------------------------------------

print("\n========================================")
print("BASIC INFORMATION")
print("========================================")

print("Number of rows:", len(data))
print("Number of columns:", len(data.columns))


print("\nColumn names:")

for column in data.columns:
    print(" -", column)


# ----------------------------------------
# 4. Show first few rows
# ----------------------------------------

print("\n========================================")
print("FIRST 5 ROWS")
print("========================================")

print(data.head())


# ----------------------------------------
# 5. Check data types
# ----------------------------------------

print("\n========================================")
print("DATA TYPES")
print("========================================")

print(data.dtypes)


# ----------------------------------------
# 6. Find possible emotion column
# ----------------------------------------

print("\n========================================")
print("POSSIBLE EMOTION COLUMNS")
print("========================================")

possible_emotion_columns = []

for column in data.columns:

    column_lower = column.lower()

    if (
        "emotion" in column_lower
        or "label" in column_lower
        or "class" in column_lower
        or "expression" in column_lower
    ):
        possible_emotion_columns.append(column)


if possible_emotion_columns:

    for column in possible_emotion_columns:

        print("\nColumn:", column)

        print(
            data[column]
            .value_counts(dropna=False)
        )

else:

    print("No obvious emotion column found.")


# ----------------------------------------
# 7. Check missing values
# ----------------------------------------

print("\n========================================")
print("MISSING VALUES")
print("========================================")

missing_values = data.isnull().sum()

print(
    missing_values[
        missing_values > 0
    ]
)


# ----------------------------------------
# 8. Dataset memory usage
# ----------------------------------------

print("\n========================================")
print("MEMORY USAGE")
print("========================================")

memory_usage = data.memory_usage(
    deep=True
).sum()

memory_usage_mb = memory_usage / (
    1024 * 1024
)

print(
    f"Approximate memory usage: "
    f"{memory_usage_mb:.2f} MB"
)


# ----------------------------------------
# 9. Final summary
# ----------------------------------------

print("\n========================================")
print("INSPECTION COMPLETE")
print("========================================")

print(
    "Rows:",
    len(data)
)

print(
    "Columns:",
    len(data.columns)
)

print("\nNext step:")
print(
    "We will determine how to convert "
    "this CSV into training images."
)