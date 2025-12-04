import os, sys
# Ensure the `src` folder is on the python path so we can import our local modules
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
from gui.app import APLEmulator

# Initialize emulator (DLL path isn't needed for this test - set to nonexistent)
em = APLEmulator(dll_path='nonexistent.dll')

# Test cases
expr_ascii = 'Result <- Input +.x Weights'
expr_unicode = 'Result ← Input +.× Weights'
expr_phi = 'Result <- phi Input'

input_matrices = """
A:
1 2 3
4 5 6

W:
1 0
0 1
1 1
"""

single = '1 2 3 4'

print('=== Test: ASCII operator and two matrices ===')
print('Expression:', expr_ascii)
print('Input:', input_matrices)
print('Result:', em.execute_apl_expression(expr_ascii, input_matrices))

print('\n=== Test: Unicode operator variant ===')
print('Expression (escaped):', expr_unicode.encode('unicode_escape').decode('ascii'))
print('Result:', em.execute_apl_expression(expr_unicode, input_matrices))

print('\n=== Test: Single matrix input -> random W generation ===')
print('Expression:', expr_ascii)
print('Input (single):', single)
print('Result:', em.execute_apl_expression(expr_ascii, single))

print('\n=== Test: Transpose (phi) ===')
print('Expr (escaped):', expr_phi.encode('unicode_escape').decode('ascii'))
print('Result:', em.execute_apl_expression(expr_phi, '1 2 3\n4 5 6'))
