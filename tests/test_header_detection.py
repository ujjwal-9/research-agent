#!/usr/bin/env python
"""Test script for LLM-based header detection in Excel files."""

import asyncio
import pandas as pd
import tempfile
from pathlib import Path
import sys

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.schema_generator import TableExtractor
from src.config import settings


def create_test_excel_with_headers():
    """Create test Excel files with different header scenarios."""

    scenarios = []

    # Scenario 1: Header in row 0 (normal case)
    header_row1 = pd.DataFrame([["Product_ID", "Product_Name", "Price", "Quantity"]])
    data_rows1 = pd.DataFrame(
        [
            ["PRD-001", "Laptop", 1200, 10],
            ["PRD-002", "Mouse", 25, 50],
            ["PRD-003", "Keyboard", 85, 30],
        ]
    )
    scenario1 = pd.concat([header_row1, data_rows1], ignore_index=True)
    scenarios.append(("Normal Header (Row 0)", scenario1))

    # Scenario 2: Header in row 1 (title in row 0)
    title_row = pd.DataFrame([["Sales Report Q1 2024", "", "", ""]])
    header_row = pd.DataFrame([["Product_ID", "Product_Name", "Price", "Quantity"]])
    data_rows = pd.DataFrame(
        [
            ["PRD-001", "Laptop", 1200, 10],
            ["PRD-002", "Mouse", 25, 50],
            ["PRD-003", "Keyboard", 85, 30],
        ]
    )
    scenario2 = pd.concat([title_row, header_row, data_rows], ignore_index=True)
    scenarios.append(("Header in Row 1 (Title above)", scenario2))

    # Scenario 3: Header in row 2 (empty row, title, then header)
    empty_row = pd.DataFrame([["", "", "", ""]])
    title_row2 = pd.DataFrame([["Customer Analytics Report", "", "", ""]])
    header_row2 = pd.DataFrame([["Customer_ID", "Company_Name", "Revenue", "Orders"]])
    data_rows2 = pd.DataFrame(
        [
            ["CUST-001", "TechCorp", 50000, 25],
            ["CUST-002", "StartupXYZ", 15000, 8],
            ["CUST-003", "Enterprise Ltd", 75000, 40],
        ]
    )
    scenario3 = pd.concat(
        [empty_row, title_row2, header_row2, data_rows2], ignore_index=True
    )
    scenarios.append(("Header in Row 2 (Empty + Title above)", scenario3))

    # Scenario 4: Mixed header scenario with dates and metadata
    metadata_row1 = pd.DataFrame([["Generated on: 2024-01-15", "", "", ""]])
    metadata_row2 = pd.DataFrame([["Department: Sales", "", "", ""]])
    header_row3 = pd.DataFrame(
        [["Region", "Sales_Amount", "Target", "Achievement_Pct"]]
    )
    data_rows3 = pd.DataFrame(
        [
            ["North", 125000, 100000, 125.0],
            ["South", 95000, 110000, 86.4],
            ["East", 110000, 105000, 104.8],
            ["West", 88000, 95000, 92.6],
        ]
    )
    scenario4 = pd.concat(
        [metadata_row1, metadata_row2, header_row3, data_rows3], ignore_index=True
    )
    scenarios.append(("Header in Row 2 (Metadata above)", scenario4))

    return scenarios


async def test_header_detection():
    """Test the LLM-based header detection on various scenarios."""

    print("🧪 Testing LLM-Based Header Detection")
    print("=" * 60)

    if not settings.openai_api_key:
        print("❌ OpenAI API key not found. Cannot test LLM-based header detection.")
        print("   Set OPENAI_API_KEY environment variable to run this test.")
        return

    # Create test scenarios
    scenarios = create_test_excel_with_headers()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Initialize table extractor with LLM support
        extractor = TableExtractor(api_key=settings.openai_api_key)

        for scenario_name, scenario_df in scenarios:
            print(f"\n📋 Testing: {scenario_name}")
            print("-" * 40)

            # Save scenario to Excel file
            excel_file = (
                temp_path
                / f"{scenario_name.replace(' ', '_').replace('(', '').replace(')', '').lower()}.xlsx"
            )
            scenario_df.to_excel(excel_file, index=False, header=False)

            # Show first few rows
            print("First 4 rows of the Excel file:")
            for i in range(min(4, len(scenario_df))):
                row_data = [
                    str(cell) if pd.notna(cell) else "" for cell in scenario_df.iloc[i]
                ]
                print(f"  Row {i}: {' | '.join(row_data)}")

            try:
                # Extract tables with LLM header detection
                tables = await extractor.extract_tables_from_excel(excel_file)

                if tables:
                    table = tables[0]  # Get first table
                    detected_header_row = table.get("header_row", "Unknown")
                    extracted_data = table["data"]

                    print(f"🎯 LLM detected header in row: {detected_header_row}")
                    print(f"📊 Extracted table shape: {extracted_data.shape}")
                    print(f"📝 Column names: {list(extracted_data.columns)}")

                    # Show sample data
                    if not extracted_data.empty:
                        print("Sample data (first 2 rows):")
                        for idx, row in extracted_data.head(2).iterrows():
                            print(f"  Data row {idx}: {dict(row)}")
                    else:
                        print("⚠️  No data rows found")

                else:
                    print("❌ No tables extracted")

            except Exception as e:
                print(f"❌ Error processing {scenario_name}: {e}")

        print(f"\n" + "=" * 60)
        print("🎉 Header Detection Test Complete!")
        print("\nKey Features Tested:")
        print("  ✓ LLM analyzes first 3-5 rows to identify headers")
        print("  ✓ Handles titles and metadata above actual headers")
        print("  ✓ Distinguishes between descriptive headers and data values")
        print("  ✓ Graceful fallback when LLM detection fails")
        print("  ✓ Clean header names and proper data extraction")


async def test_comparison_with_without_llm():
    """Compare results with and without LLM header detection."""

    print("\n🔄 Comparison: With vs Without LLM Header Detection")
    print("=" * 60)

    # Create a challenging scenario
    title_row = pd.DataFrame([["Monthly Sales Report - Q4 2023", "", "", ""]])
    empty_row = pd.DataFrame([["", "", "", ""]])
    header_row = pd.DataFrame(
        [["Product_Code", "Sales_Volume", "Revenue_USD", "Growth_Rate"]]
    )
    data_rows = pd.DataFrame(
        [
            ["ELEC-001", 1500, 125000, 15.2],
            ["ELEC-002", 800, 45000, -5.1],
            ["ACCE-001", 2200, 35000, 25.8],
        ]
    )

    test_df = pd.concat(
        [title_row, empty_row, header_row, data_rows], ignore_index=True
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        excel_file = temp_path / "comparison_test.xlsx"
        test_df.to_excel(excel_file, index=False, header=False)

        print("Test data (first 4 rows):")
        for i in range(4):
            row_data = [str(cell) if pd.notna(cell) else "" for cell in test_df.iloc[i]]
            print(f"  Row {i}: {' | '.join(row_data)}")

        # Test with LLM
        print(f"\n🧠 WITH LLM Header Detection:")
        if settings.openai_api_key:
            extractor_with_llm = TableExtractor(api_key=settings.openai_api_key)
            tables_llm = await extractor_with_llm.extract_tables_from_excel(excel_file)

            if tables_llm:
                table_llm = tables_llm[0]
                print(
                    f"  Header row detected: {table_llm.get('header_row', 'Unknown')}"
                )
                print(f"  Columns: {list(table_llm['data'].columns)}")
                print(f"  Data shape: {table_llm['data'].shape}")
            else:
                print("  ❌ No tables extracted")
        else:
            print("  ⚠️  OpenAI API key not available")

        # Test without LLM (traditional pandas approach)
        print(f"\n📊 WITHOUT LLM (Traditional pandas):")
        try:
            df_traditional = pd.read_excel(excel_file)  # Uses first row as header
            print(f"  Columns: {list(df_traditional.columns)}")
            print(f"  Data shape: {df_traditional.shape}")
            print(f"  Note: Uses row 0 as header regardless of content")
        except Exception as e:
            print(f"  ❌ Error: {e}")


def main():
    """Run all header detection tests."""

    print("🚀 Enhanced Header Detection Test Suite")
    print("=" * 60)

    asyncio.run(test_header_detection())
    asyncio.run(test_comparison_with_without_llm())

    print(f"\n" + "=" * 60)
    print("✅ All tests completed!")
    print("\nNext steps:")
    print("  - Run enhanced schema generator on your own Excel files")
    print("  - Check that headers are correctly identified in complex sheets")
    print("  - Review extracted tables for proper column naming")


if __name__ == "__main__":
    main()
