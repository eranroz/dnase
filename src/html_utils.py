"""
Utility functions for creating html for nice output
"""
__author__ = 'eranroz'

import numpy as np
import matplotlib.cm as cm
import matplotlib as mpl


def table_to_html_heatmap(table, row_labels=None, column_labels=None, caption=""):
    """
    Creates html
    @param caption: table caption
    @param table: data for the heatmap
    @param row_labels: labels for rows
    @param column_labels: labels for columns
    @return: table in html format
    """
    if column_labels is None:
        column_labels = np.arange(1, table.shape[1]+1).astype('str')
    if row_labels is None:
        row_labels = np.arange(1, table.shape[0]+1).astype('str')
    if caption:
        caption = '<caption>%s</caption>' % caption

    headers = '\n'.join(['<th>%s</th>' % th for th in column_labels])
    backgrounds = cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=np.min(table), vmax=np.max(table)), cmap=cm.Blues)
    mean_to_color = lambda x:  'rgb(%i, %i, %i)' % backgrounds.to_rgba(x, bytes=True)[:3]
    rows_trs = []
    for r_label, r_data in zip(row_labels, table):
        cell_description = "<td>%s</td>" % r_label
        cell_description += ''.join(['<td style="color:#fff;background:%s">%.2f</td>' % (mean_to_color(val), val)
                                     for val in r_data])
        # wrap in tr
        cell_description = '<tr>%s</tr>' % cell_description
        rows_trs.append(cell_description)

    template = """
<table style="font-size:85%;text-align:center;border-collapse:collapse;border:1px solid #aaa;" cellpadding="5" border="1">
{caption}
<tr style="font-size:larger; font-weight: bold;">
    <th></th>
    {headers}
</tr>
{rows_trs}
</table>
"""
    return template.format(**({'caption': caption,
                               'headers': headers,
                               'rows_trs': '\n'.join(rows_trs)
    }))


def list_to_ol(input_list):
    """
    Creates a html list based on input list
    @param input_list: list of items
    @return: html for list of items
    """
    return '<ol>{}</ol>'.format('\n'.join(['<li>%s</li>' % cell_type for cell_type in input_list]))