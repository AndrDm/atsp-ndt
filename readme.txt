ATSP Solver - Manipulator Movement Optimizer v.0.1.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This tool computes the optimal movement path for a manipulator 
by solving the Asymmetric Traveling Salesman Problem (ATSP) 
using a Mixed Integer Linear Programming (MILP) approach. 
The movement starts at Home (position 0) and returns to Home at the end.

The input file must be a square cost matrix in CSV format.
First row and first column represent the Home position (index 0).
Each row represents the source position, and each column represents the destination position.
The value at matrix[i][j] is the time cost (in seconds) to move from position i to position j.
Diagonal elements should be 0 (cost of staying in the same position).

Usage: milpr.exe <path_to_csv>

Supported CSV Formats (The parser supports multiple delimiters and decimal styles):

Dot-delimited, Semicolon-separated:
0.000;1.283;1.311;1.770
1.184;0.000;1.137;1.352
1.791;1.420;0.000;1.658
1.976;1.326;1.747;0.000

Dot-delimited, Comma-separated:
0.000,1.283,1.311,1.770
1.184,0.000,1.137,1.352
1.791,1.420,0.000,1.658
1.976,1.326,1.747,0.000

Dot-delimited, Tab-separated:
0.000	1.283	1.311	1.770
1.184	0.000	1.137	1.352
1.791	1.420	0.000	1.658
1.976	1.326	1.747	0.000

Comma-delimited (Euro style), Tab-separated:
0,000	1,283	1,311	1,770
1,184	0,000	1,137	1,352
1,791	1,420	0,000	1,658
1,976	1,326	1,747	0,000

Comma-delimited, Semicolon-separated (Whitespace allowed):
0,000; 1,283; 1,311; 1,770
1,184; 0,000; 1,137; 1,352
1,791; 1,420; 0,000; 1,658
1,976; 1,326; 1,747; 0,000

Example:
milpr.exe cost_matrix_3.csv 

Output:
Reading CSV cost_matrix_3.csv
CSV loaded
Sequential movement (Home > 1 > 2 > ... > Home): 6.054 seconds
Creating 12 binary variables and 4 MTZ variables...
Building objective with 12 terms...
Adding 8 degree constraints...
Adding 6 MTZ constraints...
| Solved - 00:00:01
Optimal movement: 5.479 seconds
Route: Home > 2 > 3 > 1 > Home

dependencies:
Microsoft Visual C++ Redistributable for Visual Studio 2015 and later (VC++ 14.0).

changelog:
--------------------------------------------------------------------------------
AD 2025-11-27 v.0.1.0 - build 1 (rustc 1.91.1)
Initial version
