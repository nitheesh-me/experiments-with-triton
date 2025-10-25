#!/usr/bin/env python3
"""triton2ir - CLI to run a Python module and capture or replay Triton IR

Usage examples:
	python triton2ir.py sample.py --callable print_addir
	python triton2ir.py sample.py --save-ir sample_ir.txt
	python triton2ir.py sample.py --read-ir sample_ir.txt

Description:
	The CLI loads the given Python file with ``runpy.run_path`` which executes
	the module's top-level code. Many example modules print Triton IR during
	import; others expose helper callables that produce or return IR.

	By default the tool will look up a callable (default ``print_addir``) and
	invoke it if it accepts zero arguments. Use ``--callable NAME`` to change
	which callable is used.

Save / read behavior:
	- ``--save-ir PATH``: run the module (and invoke the callable if zero-arg)
		while capturing stdout/stderr. If the invoked callable returns a
		``str`` or ``bytes`` value, that returned value is written exactly to
		PATH. Otherwise the captured text (stdout+stderr) is written.
	- ``--read-ir PATH``: skip running the module and print the contents of a
		previously saved IR file. This is useful for inspecting IR without
		requiring Triton/CUDA at view time.

Notes:
	- The Click-based CLI still defines a required MODULE positional
		argument; when using ``--read-ir`` the MODULE value is ignored.
	- For a nicer command-line UX install Click: ``pip install click``.
	- Triton/PyTorch/CUDA are required only when running modules that compile
		kernels; they are not needed to read previously saved IR files.

Examples:
	- Save captured IR text from a module that prints IR at import:
			python3 triton2ir.py sample.py --save-ir sample_ir.txt
	- Save the exact returned IR when a callable returns the IR string:
			python3 triton2ir.py examples/return_ir.py --callable return_ir --save-ir out.txt
	- Read previously saved IR without running the module:
			python3 triton2ir.py sample.py --read-ir sample_ir.txt
"""

from __future__ import annotations

import runpy
import sys
import traceback

try:
	import click
except Exception:  # pragma: no cover - runtime dependency handling
	click = None


def main(module: str, callable_name: str = 'print_addir', debug: bool = False, save_ir: str | None = None, read_ir: str | None = None):
	"""Load a python file and call a callable inside it.

	The callable is expected to trigger Triton kernel compilation and printing
	of the IR (for example, by compiling a triton.jit kernel and printing
	compiled_kernel.asm['ttir']). The callable is invoked with no args.
	"""

	# If click isn't available, fallback to a minimal arg parser so the script
	# fails more gracefully and provides a hint to install click.
	if click is None:
		# If click isn't present, we'll fall back to a minimal entry path.
		print('The Python package `click` is not installed. Please install it with:')
		print('    pip install click')
		sys.exit(2)

	# If read_ir is provided, just print the saved IR and exit without
	# executing the module (this lets users avoid re-running compilation).
	if read_ir:
		try:
			with open(read_ir, 'r', encoding='utf-8') as f:
				click.echo(f.read())
			return
		except Exception as e:
			raise click.ClickException(f'Failed to read IR file {read_ir}: {e}')

	# We'll capture import + callable invocation together when save_ir is
	# requested so we can prefer writing a returned string (if any). The
	# behavior is:
	# - If --read-ir is used, we already returned above.
	# - If --save-ir is used:
	#     * Run the module and (if callable is zero-arg) invoke it inside a
	#       stdout/stderr redirect, capturing all text.
	#     * If the callable returned a str/bytes, write that exact value to
	#       the save file; otherwise write the captured text.
	# - If --save-ir is not used, behave as before: run module (no capture)
	#   and invoke zero-arg callable (printing any returned repr).

	import inspect
	import io
	import contextlib

	module_globals = None
	obj = None
	result = None

	try:
		if save_ir:
			buf = io.StringIO()
			with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
				# Run the module under a non-"__main__" name so that any
				# ``if __name__ == '__main__'`` blocks do NOT execute. This
				# lets users provide test runners or demos that run when the
				# file is executed as a script, while the CLI imports the
				# module to inspect and call specific helpers.
				module_globals = runpy.run_path(module, run_name='__triton2ir__')
				obj = module_globals.get(callable_name)
				if obj is not None and callable(obj):
					# Only auto-invoke plain Python functions/methods. Triton's
					# @triton.jit returns a wrapper object whose __call__ is not a
					# regular Python function; attempting to call it directly raises
					# runtime errors. Use inspect.isfunction/ismethod to detect
					# plain callables and avoid calling JIT-wrapped kernels.
					is_plain = inspect.isfunction(obj) or inspect.ismethod(obj)
					if is_plain:
						sig = inspect.signature(obj)
						params = [p for p in sig.parameters.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty]
						if len(params) == 0:
							result = obj()
			captured = buf.getvalue()
			# decide what to persist
			try:
				if result is not None and isinstance(result, (str, bytes)):
					to_write = result.decode() if isinstance(result, bytes) else result
				else:
					to_write = captured
				with open(save_ir, 'w', encoding='utf-8') as f:
					f.write(to_write)
			except Exception as e:
				raise click.ClickException(f'Failed to write IR to {save_ir}: {e}')
		else:
			# no capture mode: run module and then handle callable
			module_globals = runpy.run_path(module, run_name='__triton2ir__')
			obj = module_globals.get(callable_name)

			if obj is None:
				raise click.ClickException(f'Callable "{callable_name}" not found in {module}')

			if not callable(obj):
				raise click.ClickException(f'Object "{callable_name}" in {module} is not callable')

			# Only auto-invoke plain Python functions/methods. For JIT-wrapped
			# kernels (e.g. objects returned by @triton.jit) calling them directly
			# will raise. Detect plain callables with inspect.isfunction/ismethod.
			is_plain = inspect.isfunction(obj) or inspect.ismethod(obj)
			if is_plain:
				sig = inspect.signature(obj)
				params = [p for p in sig.parameters.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty]
				if len(params) == 0:
					# If we already called it (in save_ir mode), avoid calling twice
					if result is None:
						result = obj()
					if result is not None:
						# Print repr for CLI feedback
						click.echo(repr(result))
				else:
					click.echo(f'Note: callable "{callable_name}" expects {len(params)} positional argument(s).')
					click.echo('The module was executed at import time; if it printed IR already, no further action is required.')
					click.echo('If you want to invoke the callable with arguments, run it from within the module or provide a small wrapper function that takes no args.')
			else:
				# Non-plain callables (likely JIT wrappers) are not auto-invoked.
				click.echo(f'Note: callable "{callable_name}" is not a plain Python function; skipping automatic invocation.')
				click.echo('If this is a Triton JIT kernel, call it from within the module or provide a zero-arg wrapper that returns the IR string.')
	except Exception as e:
		if debug:
			traceback.print_exc()
		raise click.ClickException(f'Failed to run module or callable {module}: {e}')


def _minimal_entry():
	"""Fallback entry used when click isn't importable and user executed file."""
	# Basic fallback to keep UX friendly if someone runs `python triton2ir.py`.
	if len(sys.argv) < 2:
		print('Usage: triton2ir.py <module_path> [--callable NAME] [--save-ir PATH] [--read-ir PATH]')
		sys.exit(2)
	# crude parsing
	module = sys.argv[1]
	callable_name = 'print_addir'
	save_ir = None
	read_ir = None
	if '--callable' in sys.argv:
		i = sys.argv.index('--callable')
		if i + 1 < len(sys.argv):
			callable_name = sys.argv[i + 1]
	if '--save-ir' in sys.argv:
		i = sys.argv.index('--save-ir')
		if i + 1 < len(sys.argv):
			save_ir = sys.argv[i + 1]
	if '--read-ir' in sys.argv:
		i = sys.argv.index('--read-ir')
		if i + 1 < len(sys.argv):
			read_ir = sys.argv[i + 1]
	if click is None:
		print('Note: install click for a richer CLI: pip install click')
	try:
		# If read_ir provided, read and print saved IR and exit
		if read_ir:
			try:
				with open(read_ir, 'r', encoding='utf-8') as f:
					print(f.read())
				sys.exit(0)
			except Exception as e:
				print(f'Failed to read IR file {read_ir}: {e}')
				sys.exit(2)

		# run module and optionally capture stdout/stderr
		if save_ir:
			import io
			import contextlib

			buf = io.StringIO()
			with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
				module_globals = runpy.run_path(module, run_name='__triton2ir__')
			captured = buf.getvalue()
			try:
				with open(save_ir, 'w', encoding='utf-8') as f:
					f.write(captured)
				print(f'Saved captured output to {save_ir}')
			except Exception as e:
				print(f'Failed to write IR to {save_ir}: {e}')
				sys.exit(2)
		else:
			module_globals = runpy.run_path(module, run_name='__main__')

		obj = module_globals.get(callable_name)
		if obj is None:
			print(f'Callable "{callable_name}" not found in {module}')
			sys.exit(2)
		# Only auto-invoke if callable accepts zero args
		import inspect
		sig = inspect.signature(obj)
		params = [p for p in sig.parameters.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty]
		if len(params) == 0:
			res = obj()
			if res is not None:
				print(repr(res))
		else:
			print(f'Note: callable "{callable_name}" expects {len(params)} positional argument(s).')
			print('The module was executed at import time; if it printed IR already, no further action is required.')
			print('If you want to invoke the callable with arguments, run it from within the module or provide a small wrapper function that takes no args.')
	except Exception:
		traceback.print_exc()
		sys.exit(1)


if __name__ == '__main__':
	# If click is available, expose a nicer CLI. Otherwise use the fallback.
	if click is not None:
		@click.command()
		@click.argument('module', type=click.Path(exists=True))
		@click.option('--callable', '-c', 'callable_name', default='print_addir', help='Callable to execute in the module')
		@click.option('--save-ir', type=click.Path(), default=None, help='Path to save captured stdout/stderr (IR) from running the module')
		@click.option('--read-ir', type=click.Path(exists=True), default=None, help='Read and print a previously saved IR file and exit without running the module')
		@click.option('--debug', is_flag=True, help='Print full traceback on error')
		def _cli(module, callable_name, save_ir, read_ir, debug):
			return main(module, callable_name, debug, save_ir=save_ir, read_ir=read_ir)

		_cli()
	else:
		_minimal_entry()

