"""
Compare_correlation_NCBI.py

 Gets the theoretical peptides generated from the sequences
 and filtered by the LCMSMS data
 and compares how many match an actual PMF within a certain tolerance
 tolerance is usually 0.2
 returns the results with the number of matches for each species
 ordered by highest number of matches
 highest number of matches should be the species most closely related
 to species in the database
"""

import sys
from pathlib import Path
import argparse
from typing import List, Tuple, Dict, Optional

import pandas as pd

################
# FUNCTIONS
################


def file_test(arg: str) -> Path:
    """Tests if the input file exists"""
    p = Path(arg)
    if p.is_file():
        return p
    else:
        raise FileNotFoundError(arg)


def directory_test(arg: str) -> Path:
    """Test if the input directory exists"""
    p = Path(arg)
    if p.is_dir():
        return p
    else:
        raise Exception(f"The input directory does not exist {p}")


def output_test(arg: str) -> Path:
    """Test if directory of new output file exists"""
    p = Path(arg)
    par = p.parent
    if par.is_dir():
        return p
    else:
        raise Exception(
            "The directory of the new output file does not exist {0}".format(p)
        )


def range_test(arg):
    """
    Tests if the input is a valid range for the mass range of a ZooMS sample.
    """
    if (
        not isinstance(arg, tuple)
        or len(arg) != 2
        or not all(isinstance(x, float) for x in arg)
    ):
        raise ValueError("The input range should in the form (min, max)")

    if arg[0] < arg[1]:
        raise ValueError(
            """The input range is invalid. 
            The min value should be smaller than the max value (min, max)"""
        )


# test if --top5 input is 0 or 1 only
def test_01(arg):
    arg = int(arg)
    if arg == 0 or arg == 1:
        return arg
    else:
        raise Exception("The input -m5 --top5 should be the integer 0 or 1 only")


def parse_args(argv):
    description = """This code compares the experimental PMF peaks from a sample
    and the theoretical peaks generated from the all the species COL1 theoretical 
    peptides generated in the theoretical peptides pipeline. It will score a match 
    if the theoretical peptide m/z valus is within a specified threshold (+- 0.2 is default).
    The output to the command line will be the top 10 match scores (number of matches) species.
    All species match scores wil be outputted to a CSV file. """

    # set up argparse
    parser = argparse.ArgumentParser(description=description)
    # add arguments
    # adds the folder where the input theoretical peak spectrums are
    parser.add_argument(
        "-it",
        "--inputTheor",
        help="the folder that contains the theoretical peptides csv files to compare against PMF",
        type=directory_test,
        required=True
    )
    # adds where the output file should be saved
    parser.add_argument(
        "-o",
        "--output",
        help="The file name for the output results file of number of matches",
        type=output_test,
        required=True
    )
    # adds where the input peptide mass fingerprint
    parser.add_argument(
        "-ip",
        "--inputPMF",
        help="The input Peptide mass fingerprint (PMF) from an unknown organism.",
        type=file_test,
        required=True
    )
    parser.add_argument(
        "-t",
        "--threshold",
        help="The threshold for matches between the experimental and theoretical spectrum. Default is 0.2 Da",
        default=0.2,
        type=float,
    )
    parser.add_argument(
        "-mr",
        "--mass_range",
        help="""The mass (m/z) range within which a match can occur. The format must be (min,max) with
        the first value the minimum value a match can be and the second value the maximum value a match can be.
        Default is (800,3500) which is 800-3,500 m/z""",
        default=(800.0, 3500.0),
        type=range_test
    )
    parser.add_argument(
        "-m5",
        "--top5",
        help="""If 1 is inputted will provide an excel file of m/z peak matches for the top 5 match counts as an xlsx (excel) file.
                        Default is 0""",
        default=0,
        type=test_01,
    )
    return parser.parse_args(argv)


def read_exp_pmf(input_pmf: Path, mass_range: Tuple[float, float]) -> pd.DataFrame:
    """
    Reads in the experimental PMF text file with m/z values.
    
    Args:
        input_pmf: Path to the experimental PMF file.
        mass_range: Tuple of (min, max) mass values to filter.
        
    Returns:
        DataFrame containing filtered experimental peaks.
    """
    dtype = {"MZ": "float32", "intensity": "float32"}
    try:
        df = pd.read_table(
            input_pmf, sep="\t", header=None, names=["MZ", "intensity"], dtype=dtype
        )
    except Exception:
        raise Exception(f"Could not read {input_pmf}. Ensure it is a tab-separated file.")

    # Filter by mass range
    df = df[df["MZ"].between(mass_range[0], mass_range[1])].reset_index(drop=True)
    return df


def load_theoretical_data(input_dir: Path, mass_range: Tuple[float, float]) -> List[pd.DataFrame]:
    """
    Reads all theoretical peptide CSV files from the specified directory.
    
    Args:
        input_dir: Directory containing CSV files.
        mass_range: Tuple of (min, max) mass values to filter.
        
    Returns:
        List of DataFrames, one per species file.
    """
    csv_files = list(input_dir.glob("*.csv"))
    
    dtype = {
        "mass1": "float32",
        "GENUS": "category",
        "SPECIES": "category",
        "pep_seq": "category",
    }
    usecols = [
        "mass1",
        "genus",
        "species",
        "subfamily",
        "family",
        "order",
        "pep_seq",
        "pep_start",
        "pep_end",
        "hyd_count",
        "deam_count",
        "missed_cleaves",
    ]
    
    df_list = []
    for csv in csv_files:
        try:
            df = pd.read_csv(csv, sep=",", dtype=dtype, usecols=usecols)
            # Filter by mass range immediately to save memory/processing
            df = df[df["mass1"].between(mass_range[0], mass_range[1])].reset_index(drop=True)
            if not df.empty:
                df_list.append(df)
        except Exception as e:
            print(f"Warning: Failed to read {csv}: {e}", file=sys.stderr)
            
    return df_list


def compare_peaks(
    theor_df: pd.DataFrame, 
    exp_df: pd.DataFrame, 
    threshold: float
) -> Tuple[int, pd.DataFrame]:
    """
    Compares one set of theoretical peptides against experimental peaks.
    
    Args:
        theor_df: DataFrame of theoretical peptides.
        exp_df: DataFrame of experimental peaks.
        threshold: Mass tolerance for matching.
        
    Returns:
        Tuple containing:
            - match_count (int): Number of experimental peaks matched.
            - matches_df (DataFrame): Detailed matches.
    """
    # Create copies to avoid modifying original dataframes
    theor = theor_df.copy()
    exp = exp_df.copy()
    
    # Add temporary join key for cross-product
    theor["_join"] = 1
    exp["_join"] = 1
    
    # Calculate match bounds
    exp["_mz_min"] = exp["MZ"] - threshold
    exp["_mz_max"] = exp["MZ"] + threshold
    
    # Perform cross join
    merged = pd.merge(exp, theor, on="_join", how="inner")
    
    # Filter for matches within tolerance
    matches = merged[
        (merged["mass1"] >= merged["_mz_min"]) & 
        (merged["mass1"] <= merged["_mz_max"])
    ].copy()
    
    # If an experimental peak matches multiple theoretical peptides, 
    # we count it as one match (the peak is explained).
    unique_matches = matches.drop_duplicates(subset=["MZ"])
    match_count = len(unique_matches)
    
    # Clean up temporary columns
    unique_matches = unique_matches.drop(columns=["_join", "_mz_min", "_mz_max"])
    
    return match_count, unique_matches


def process_all_species(
    theor_dfs: List[pd.DataFrame],
    exp_df: pd.DataFrame,
    threshold: float,
    total_peaks: int
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Runs the comparison for all loaded theoretical species.
    
    Args:
        theor_dfs: List of theoretical peptide DataFrames.
        exp_df: Experimental peaks DataFrame.
        threshold: Matching threshold.
        total_peaks: Total number of peaks in experimental data (for reference).
        
    Returns:
        Tuple containing:
            - summary_df: DataFrame with match counts for all species.
            - matches_map: Dictionary mapping species name to its matches DataFrame.
    """
    results = []
    matches_map = {}
    
    for theor_df in theor_dfs:
        if theor_df.empty:
            continue
            
        # Extract taxonomy info from the first row
        # Assuming all rows in a file belong to the same species
        first_row = theor_df.iloc[0]
        species_name = str(first_row["species"])
        
        match_count, matches_df = compare_peaks(theor_df, exp_df, threshold)
        
        # Store summary info
        summary_entry = {
            "species": species_name,
            "genus": first_row["genus"],
            "subfamily": first_row["subfamily"],
            "family": first_row["family"],
            "order": first_row["order"],
            "Match": match_count
        }
        results.append(summary_entry)
        
        # Store detailed matches
        matches_map[species_name] = matches_df
        
    if not results:
        return pd.DataFrame(), {}
        
    summary_df = pd.DataFrame(results)
    summary_df["Maximum Possible"] = total_peaks
    
    # Sort by matches descending
    summary_df = summary_df.sort_values(by="Match", ascending=False).reset_index(drop=True)
    
    # Reorder columns to match original output preference if possible
    cols = ["genus", "species", "Match", "Maximum Possible", "subfamily", "family", "order"]
    # Filter cols that exist
    cols = [c for c in cols if c in summary_df.columns]
    summary_df = summary_df[cols]
    
    return summary_df, matches_map


def save_top_matches(
    matches_map: Dict[str, pd.DataFrame], 
    summary_df: pd.DataFrame, 
    output_file: Path
):
    """
    Saves the detailed matches for the top 5 species to an Excel file.
    
    Args:
        matches_map: Dictionary of matches per species.
        summary_df: Ranked summary DataFrame.
        output_file: Path to the main output CSV (used to determine Excel path).
    """
    # Get top 5 species names
    top_species = summary_df.head(5)["species"].tolist()
    
    output_dir = output_file.parent
    excel_path = output_dir / "top5_matches.xlsx"
    
    print(f"\nSaving top 5 matches details to: {excel_path}")
    
    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        for i, species in enumerate(top_species):
            if species in matches_map:
                df = matches_map[species]
                # Rename columns for report
                df = df.rename(columns={
                    "MZ": "Exp MZ",
                    "intensity": "Exp intensity",
                    "mass1": "Theor MZ"
                })
                
                # Sheet name (limited length)
                sheet_name = f"Rank_{i+1}"
                df.to_excel(writer, sheet_name=sheet_name, index=False)


def main(argv=sys.argv[1:]):
    """Main method and logic"""
    print("""####################
Program: Compare_NCBI.py
#####################""")

    args = parse_args(argv)

    # 1. Load Experimental Data
    input_pmf_path = Path(args.inputPMF)
    try:
        exp_df = read_exp_pmf(input_pmf_path, args.mass_range)
        total_peaks = len(exp_df)
        print(f"Loaded {total_peaks} experimental peaks from {input_pmf_path.name}")
    except Exception as e:
        print(f"Error loading experimental PMF: {e}")
        sys.exit(1)

    # 2. Load Theoretical Data
    input_theor_dir = Path(args.inputTheor)
    theor_dfs = load_theoretical_data(input_theor_dir, args.mass_range)
    print(f"Loaded theoretical data for {len(theor_dfs)} species.")

    # 3. Run Comparison
    print(f"\nThreshold for match is +- {args.threshold}")
    summary_df, matches_map = process_all_species(
        theor_dfs, exp_df, args.threshold, total_peaks
    )

    if summary_df.empty:
        print("No matches found or no theoretical data available.")
        sys.exit(0)

    # 4. Output Results
    print("RESULTS:")
    print(summary_df.head(10).to_markdown())
    
    output_path = Path(args.output)
    summary_df.to_csv(output_path)
    print(f"\nFull results saved to {output_path}")

    # 5. Save Top 5 Details if requested
    if args.top5 == 1:
        save_top_matches(matches_map, summary_df, output_path)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
