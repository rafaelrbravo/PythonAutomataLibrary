"""Characterize PAL's source-line diagnostics for typed and untyped compiled calls."""
import pytest

# Marker names recognized by PAL's annotation-based AST transformer.
PopGrid = object
PDEgrid = object


def test_annotated_popgrid_compiled_error_reports_line(api, safe_mode):
    if not safe_mode:
        pytest.skip('Fast mode omits validation diagnostics')
    g = api.NewPopGrid((4,), capacity=10)

    @api.njit
    def invalid(grid: PopGrid):
        grid.Add(-1, 0)
        grid.Update()

    with pytest.raises(ValueError, match='source line'):
        invalid(g)


def test_annotated_pde_compiled_error_reports_line(api, safe_mode):
    if not safe_mode:
        pytest.skip('Fast mode omits validation diagnostics')
    g = api.NewPDEgrid((5,))
    g.SetTimeSpaceStep(1.0, 1.0)

    @api.njit
    def invalid(grid: PDEgrid):
        grid.Diffusion(0.6)

    with pytest.raises(ValueError, match='source line'):
        invalid(g)


@pytest.mark.xfail(strict=True, reason='Unannotated PAL grid arguments are not recognized for source-line AST diagnostics')
def test_unannotated_popgrid_compiled_error_reports_line(api, safe_mode):
    if not safe_mode:
        pytest.skip('Fast mode omits validation diagnostics')
    g = api.NewPopGrid((4,), capacity=10)

    @api.njit
    def invalid(grid):
        grid.Add(-1, 0)
        grid.Update()

    with pytest.raises(ValueError, match='source line'):
        invalid(g)


@pytest.mark.xfail(strict=True, reason='Unannotated PAL grid arguments are not recognized for source-line AST diagnostics')
def test_unannotated_pde_compiled_error_reports_line(api, safe_mode):
    if not safe_mode:
        pytest.skip('Fast mode omits validation diagnostics')
    g = api.NewPDEgrid((5,))
    g.SetTimeSpaceStep(1.0, 1.0)

    @api.njit
    def invalid(grid):
        grid.Diffusion(0.6)

    with pytest.raises(ValueError, match='source line'):
        invalid(g)
