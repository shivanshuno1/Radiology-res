import pandas as pd


def show_template_examples(train):

    grouped = train.groupby("template_content")

    print(f"\nTOTAL UNIQUE TEMPLATES: {len(grouped)}")

    for i, (template, group) in enumerate(grouped, start=1):

        row = group.iloc[0]

        print("\n" + "=" * 100)
        print(f"TEMPLATE {i}")
        print("=" * 100)

        print(f"\nCASE ID:")
        print(row["case_id"])

        print(f"\nMODALITY:")
        print(row["modality"])

        print(f"\nBODY PART:")
        print(row["body_part"])

        print("\n--- TEMPLATE ---")
        print(row["template_content"])

        print("\n--- DICTATION ---")
        print(row["dictation"])

        print("\n--- EXPECTED REPORT ---")
        print(row["report"])


if __name__ == "__main__":

    train = pd.read_csv("data/train.csv")

    show_template_examples(train)