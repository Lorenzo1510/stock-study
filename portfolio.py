import sys
sys.path.append(r'C:\Users\loren\OneDrive\Desktop\Workbench\stock-project')

from src.etl.etl import PredETL

pred = PredETL()
pred.process()