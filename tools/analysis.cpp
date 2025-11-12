#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include <cstddef>

int main(int argc, char **argv) {
    // Load an MLIR file
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(llvm::MemoryBuffer::getFileOrSTDIN(argv[1]), llvm::SMLoc());

    mlir::MLIRContext context;
    mlir::OwningOpRef module = mlir::parseSourceFile(sourceMgr, &context);

    if (module) {
        // Module has been parsed successfully, now you can analyze
        module->dump();
    } else {
        llvm::errs() << "Error parsing MLIR file\n";
    }

    return 0;
}
