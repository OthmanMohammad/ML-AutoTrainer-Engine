import pandas as pd
from io import StringIO
from src.data_processing import read_csv

def test_read_csv():
    # Given: A sample CSV in string format
    csv_content = "a,b,c\n1,2,3\n4,5,6"
    csv_file = StringIO(csv_content)

    # When: The read_csv function is invoked
    df = read_csv(csv_file)

    # Then: We expect a dataframe with the provided content
    assert df.shape == (2, 3)  # 2 rows, 3 columns
    assert (df.columns == ["a", "b", "c"]).all()
    assert df.iloc[0, 0] == 1  # first row, first column value
    assert df.iloc[1, 2] == 6  # second row, third column value
