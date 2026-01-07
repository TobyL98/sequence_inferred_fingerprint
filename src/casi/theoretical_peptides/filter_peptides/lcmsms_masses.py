"""
peptide_rules.py
Code uses the LC-MSMS data from multiple species collagen A1 and A2 
It groups the data by peptides that have the same start and end positions.
It uses these groups to work out common oxidation and deamidations in the peptide.
These can then be applied to theoretically (in silico) generated peptides
to predict what likely PTMs are in that peptide
OUTPUTS: sequence_masses.csv that is used in integrate.py
"""

import re
from pathlib import Path
import sys
from collections import namedtuple
from typing import List, Set, Tuple

import pandas as pd

Positions = namedtuple(
    "Positions",
    [
        "start",
        "end",
        "length",
        "start_list",
        "end_list",
    ]
)

def mod_count(mods: str, res: str) -> int:
    """Calculates the number of PTMs in targeted residues (P, K, N/Q).

    Based on the Pep-var_mod string from LC-MS/MS data.

    Args:
        mods (str): The modification string (e.g., "Oxidation (K); 2 Oxidation (P)").
        res (str): Regex pattern for the residue to count (e.g., "[MPK]" or "NQ").

    Returns:
        int: Total count of modifications for the target residues.
    """
    mods = str(mods)
    total_count = 0
    
    # Split modifications by semicolon if multiple exist
    mod_entries = [m.strip() for m in mods.split(";")] if ";" in mods else [mods.strip()]

    for entry in mod_entries:
        # Check if the modification affects the target residue
        if re.search(res, entry):
            # Default count is 1
            count = 1
            
            # Check if there is a multiplier (e.g., "2 Oxidation (P)")
            # The number is usually at the start of the string
            match = re.match(r"(\d+)", entry)
            if match:
                count = int(match.group(1))
            
            total_count += count

    return total_count


def data_load(dirpath: Path) -> pd.DataFrame:
    """Loads in all the LCMSMS mascot csv files from the LCMSMS folder.

    Concatenates them into one single dataframe.

    Args:
        dirpath (Path): The path to the directory that contains the LCMSMS csv files.

    Returns:
        pd.DataFrame: Contains all the LCMSMS data in one dataframe.
    """
    csv_files = list(dirpath.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {dirpath}")
        return pd.DataFrame()

    df_list = []
    dtypes = {"pep_score": "float32", "pep_exp_mr": "float32"}
    use_cols = [
        "pep_seq",
        "pep_score",
        "pep_start",
        "pep_end",
        "pep_exp_mr",
        "pep_miss",
        "pep_var_mod",
        "prot_acc",
    ]
    for csv in csv_files:
        try:
            new_df = pd.read_csv(csv, sep=",", dtype=dtypes, usecols=use_cols)
            # drop all empty rows
            new_df.dropna(how="all", inplace=True)
            # Ensure start/end are present and integer
            new_df = new_df.dropna(subset=["pep_start", "pep_end"])
            new_df = new_df.astype({"pep_start": int, "pep_end": int})
            df_list.append(new_df)
        except Exception as e:
            print(f"Error reading {csv}: {e}")

    if not df_list:
        return pd.DataFrame()

    # combines all dataframes in the list into single df
    df = pd.concat(df_list)

    lcmsms_df = df.sort_values(by=["pep_start", "pep_end"])
    return lcmsms_df

def find_positions(row: pd.Series) -> Positions:
    """Finds the start, end position and length for each peptide fragment.

    The start and end position are the start and end positions in the full protein sequence.
    Each row is a different peptide fragment. Creates start and end lists
    that allow for frameshifts in different peptides.

    Args:
        row (pd.Series): Row from the LC-MS/MS peptide fragments dataframe.

    Returns:
        Positions: Contains the start position, end positions, fragment sequence length,
            start list and end list.
    """

    pep_start = int(row["pep_start"])
    pep_end = int(row["pep_end"])
    seq_length = pep_end - pep_start

    # allows four behind and four after to account for frameshifts
    start_list = list(range(pep_start - 4, pep_start + 4))
    # creates an end list
    end_list = list(range(pep_end - 4, pep_end + 4))
    
    positions = Positions(
        pep_start,
        pep_end,
        seq_length,
        start_list,
        end_list
    )

    return positions

def get_peptide_subset(df: pd.DataFrame, positions: Positions) -> pd.DataFrame:
    """Filters the dataframe for peptides matching the position window and length.

    This effectively finds frameshifted versions of the same peptide.

    Args:
        df (pd.DataFrame): The full LC-MS/MS dataframe.
        positions (Positions): The position window and length to filter by.

    Returns:
        pd.DataFrame: A subset of the dataframe matching the criteria.
    """
    subset = df.loc[
        df["pep_start"].isin(positions.start_list)
        & df["pep_end"].isin(positions.end_list)
        & ((df["pep_end"] - df["pep_start"]) == positions.length)
    ].copy()
    return subset

def process_peptide_subset(subset_df: pd.DataFrame, seq_id: int) -> pd.DataFrame:
    """Processes a subset of peptides: deduplicates, calculates PTMs, and aggregates.

    Args:
        subset_df (pd.DataFrame): The subset of peptides to process.
        seq_id (int): A unique identifier for the peptide sequence group.

    Returns:
        pd.DataFrame: The aggregated dataframe with calculated PTM counts and predicted PMF.
    """
    # Keep top pep_score for each modification variant
    subset_df = subset_df.sort_values(by=["pep_score"], ascending=False)
    subset_df = subset_df.drop_duplicates(
        subset=[
            "pep_seq",
            "pep_start",
            "pep_end",
            "pep_miss",
            "pep_var_mod",
            "prot_acc",
        ]
    )
    subset_df = subset_df.sort_values(by=["prot_acc"])
    subset_df = subset_df.reset_index(drop=True)
    
    subset_df["pep_id"] = seq_id

    # Calculate modification counts
    subset_df["hyd_count"] = subset_df["pep_var_mod"].apply(mod_count, res=r"[MPK]")
    subset_df["deam_count"] = subset_df["pep_var_mod"].apply(mod_count, res=r"NQ")

    # Group by peptide properties to merge protein accessions and aggregate scores/masses
    aggregated_df = subset_df.groupby(
        ["pep_seq", "pep_miss", "hyd_count", "deam_count", "pep_id"],
        as_index=False,
        dropna=False,
    ).agg(
        {
            "prot_acc": ", ".join,
            "pep_score": "max",
            "pep_exp_mr": "mean",
            "pep_start": "min",
            "pep_end": "min",
        }
    )
    
    # Calculate predicted PMF (mass + proton)
    aggregated_df["PMF_predict"] = aggregated_df["pep_exp_mr"] + 1.0
    
    return aggregated_df

def generate_peptide_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Main logic to group and filter peptides from the raw LCMSMS data.

    Iterates through rows, finds groups of similar peptides (frameshifts),
    and aggregates their PTM information.

    Args:
        df (pd.DataFrame): The raw LC-MS/MS dataframe.

    Returns:
        pd.DataFrame: The processed dataframe containing peptide rules.
    """
    processed_positions: Set[Tuple[int, int, int]] = set()
    pep_seq_df_list = []
    seq_count = 0

    for _, row in df.iterrows():
        positions = find_positions(row)
        
        # Check if this specific start/end/length combo has been handled
        current_pos_key = (positions.start, positions.end, positions.length)
        
        if current_pos_key not in processed_positions:
            seq_count += 1
            
            # Get all related peptides (frameshifts)
            subset_df = get_peptide_subset(df, positions)
            
            if not subset_df.empty:
                processed_subset = process_peptide_subset(subset_df, seq_count)
                pep_seq_df_list.append(processed_subset)
            
            # Mark all covered positions as processed to avoid redundancy
            # We iterate through the start_list and end_list in parallel
            # assuming they represent the frameshifted windows
            for s, e in zip(positions.start_list, positions.end_list):
                processed_positions.add((s, e, positions.length))

    if not pep_seq_df_list:
        return pd.DataFrame()

    # add all the different peptide dfs into one df
    all_peps_df = pd.concat(pep_seq_df_list)
    return all_peps_df


def mass_lcsmsms(lcmsms_dir, output_folder):
    """Generates LCMSMS filter rules from raw data and saves to CSV.

    Args:
        lcmsms_dir (Path): Directory containing the LCMSMS data CSV files.
        output_folder (Path): Directory where the output CSV will be saved.

    Raises:
        FileNotFoundError: If lcmsms_dir does not exist.
    """
    print("Generating LCMSMS Filter Rules")
    if not lcmsms_dir.is_dir():
        raise FileNotFoundError(f"""The directory containing the LCMSMS data does not exist {lcmsms_dir}.
Ensure the following directory is created and put the LCMSMS data in it""")
    
    # read LCMSMS CSV files and merge to one dataframe
    all_df = data_load(lcmsms_dir)
    
    if all_df.empty:
        print("No data loaded from LCMSMS directory.")
        return

    # filter to obtain desired output of likely PTMs
    final_peps_df = generate_peptide_rules(all_df)

    if final_peps_df.empty:
        print("No peptides generated after filtering.")
        return

    # reorder for presentation
    correct_order = [
        "pep_id",
        "pep_seq",
        "pep_start",
        "pep_end",
        "pep_exp_mr",
        "hyd_count",
        "deam_count",
        "pep_miss",
        "prot_acc",
        "pep_score",
        "PMF_predict",
    ]
    final_peps_df = final_peps_df.reindex(columns=correct_order)
    # Reset index to preserve the original index behavior (creates "index" column)
    final_peps_df = final_peps_df.reset_index()

    # save as a csv file
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / "lcmsms_masses.csv"
    final_peps_df.to_csv(output_file, sep=",")
    print(f"Output: {output_file}")
    print("######################################")


if __name__ == "__main__":
    sys.exit()
