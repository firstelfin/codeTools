#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   cliCommand.py
@Time    :   2024/12/11 11:49:10
@Author  :   firstElfin 
@Version :   1.0
@Desc    :   None
'''

PARAM_INSTANCE = """
Instance1:
```shell

```
"""

def shell_get_terminal_parameters(shell_parse):
    if shell_parse.param:
        print(
        )
    shell_parse.add_argument('-t', '--terminal', action='store_true', help='Get terminal parameters')
    return shell_parse
