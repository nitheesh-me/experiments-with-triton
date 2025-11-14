"""sample for parsing ttg blocked layout"""
from pathlib import Path
import mlir
from mlir.dialect import DialectType, Dialect
# from mlir.astnodes import IntegerLiteral, ArrayLiteral, DictLiteral

class BlockedAttr(DialectType):
    """Represents the attribute #ttg.blocked<{ … }>."""
    _syntax_ = (
        '#ttg.blocked<'
          '{sizePerThread = '   '{sizePerThread.array_literal}, '
           'threadsPerWarp = '  '{threadsPerWarp.array_literal}, '
           'warpsPerCTA = '     '{warpsPerCTA.array_literal}, '
           'order = '           '{order.array_literal}'
          '}>'
    )
    _fields_ = ['sizePerThread', 'threadsPerWarp', 'warpsPerCTA', 'order']

    def __init__(self, sizePerThread, threadsPerWarp, warpsPerCTA, order):
        self.sizePerThread   = sizePerThread    # list of int
        self.threadsPerWarp  = threadsPerWarp
        self.warpsPerCTA     = warpsPerCTA
        self.order           = order

    def dump(self, indent=0):
        return (
            f"#ttg.blocked<{{"
            f"sizePerThread = [{','.join(map(str,self.sizePerThread))}], "
            f"threadsPerWarp = [{','.join(map(str,self.threadsPerWarp))}], "
            f"warpsPerCTA = [{','.join(map(str,self.warpsPerCTA))}], "
            f"order = [{','.join(map(str,self.order))}]"
            f"}}>"
        )

# Register the dialect
ttg = Dialect(
    name='ttg',
    types=[BlockedAttr],
    ops=[],
)


# attr = BlockedAttr(sizePerThread=[1], threadsPerWarp=[32], warpsPerCTA=[4], order=[0])
# print(attr.dump())

ast = mlir.parse_path(Path("tools/sample_gpu.mlir"), dialects=[ttg])
print(ast.dump())