import pandas as pd

class excel_ws(openpyxl.worksheet.worksheet.Worksheet):
    def load_df(self, df):
        self.nrows = df.shape[0] + 1
        self.ncols = df.shape[1] + 1
        return self

    def get_row(self, i):
        return self[i:i]

    def get_col(self, i):
        j = get_column_letter(i)
        return self[f"{j}:{j}"]
    
    def format_cols(self, start, end, format_dict):
        if start is None:
            start = 1
        if end is None:
            end = self.ncols
        for i in range(start, end + 1):
            col = self.get_col(i)
            for cell in col:
                for j1, j2 in format_dict.items():
                    setattr(cell, j1, j2)

    def format_row(self, start, end, format_dict):
        if start is None:
            start = 1
        if end is None:
            end = self.nrows
        for i in range(start, end + 1):
            row = self.get_row(i)            
            for cell in row:
                for j1, j2 in format_dict.items():
                    setattr(cell, j1, j2)
    @staticmethod
    def format_str(i, j):
        return get_column_letter(j) + str(i)

    def cond_format(self, p0, p1, cond_format):
        s0 = self.format_str(p0[0], p0[1])
        s1 = self.format_str(p1[0], p1[1])
        srange = s0 + ":" + s1
        print(srange)
        self.conditional_formatting.add(srange, cond_format)

    def cond_format_cols(self, start, end, cond_format):
        if start is None:
            start = 1
        if end is None:
            end = self.ncols
        self.cond_format((1, start), (self.nrows, end), cond_format)

    def cond_format_cols_list(self, cols, cond_format, start_row=None, end_row=None):
        if start_row is None:
            start_row = 1
        if end_row is None:
            end_row = self.nrows
        for i in cols:
            self.cond_format((start_row, i), (end_row, i), cond_format)
        
conditional_format = ColorScaleRule(
    start_type='num',
    start_value= -0.1,
    start_color='00FF0000',
    mid_type='num',
    mid_value=0,
    mid_color='00FFFFFF',
    end_type='num',
    end_value= 0.1,
    end_color='0000FF00'
)        
    
rule = DataBarRule(start_type='num', start_value=10, end_type='num', end_value='100',
                   color="FF638EC6", showValue="None", minLength=None, maxLength=None)





with pd.ExcelWriter("test.xlsx", engine='openpyxl') as writer:
    l_df.to_excel(writer, "test")
    ws = writer.book.get_sheet_by_name("test")    
    ws.__class__ = excel_ws
    ws.load_df(l_df)    
    ws.cond_format_cols_list(cols=range(2, 1 + l_df.shape[1], 2),
                             start_row=2, 
                             cond_format= conditional_format)
    ws.cond_format_cols_list(cols=range(3, 2 + l_df.shape[1], 2),
                             start_row=2,
                             cond_format= rule)
    



from openpyxl.formatting.rule import ColorScaleRule, DataBarRule, DataBar
from openpyxl.formatting.rule import Rule
pd.Series(dir(ws))

93                    columns
94                cond_format
95           cond_format_cols
96     conditional_formatting

for range_string, i in ws.conditional_formatting._cf_rules.items():
    break

