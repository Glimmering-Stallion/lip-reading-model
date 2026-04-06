
import os

def to_head(projectpath, border_pt: int = 8):
    pathlayers = os.path.join(projectpath, "layers/").replace("\\", "/")
    return r"""
\documentclass[border=""" + str(border_pt) + r"""pt, multi, tikz]{standalone} 
\usepackage{import}
\subimport{"""+ pathlayers + r"""}{init}
\usetikzlibrary{positioning}
\usetikzlibrary{calc}
\usetikzlibrary{3d} %for including external image 
"""

def to_cor():
    return r"""
\def\ConvColor{rgb:yellow,5;red,2.5;white,5}
\def\ConvReluColor{rgb:yellow,5;red,5;white,5}
\def\PoolColor{rgb:red,1;black,0.3}
\def\UnpoolColor{rgb:blue,2;green,1;black,0.3}
\def\FcColor{rgb:blue,5;red,2.5;white,5}
\def\FcReluColor{rgb:blue,5;red,5;white,4}
\def\SoftmaxColor{rgb:magenta,5;black,7}   
\def\SumColor{rgb:blue,5;green,15}
\def\TensorIoColor{rgb:black,3.5;white,14}
"""

def to_begin():
    return r"""
\newcommand{\copymidarrow}{\tikz \draw[-Stealth,line width=0.8mm,draw={rgb:blue,4;red,1;green,1;black,3}] (-0.3,0) -- ++(0.3,0);}

\begin{document}
\begin{tikzpicture}
\tikzstyle{connection}=[ultra thick,every node/.style={sloped,allow upside down},draw=\edgecolor,opacity=0.7]
\tikzstyle{copyconnection}=[ultra thick,every node/.style={sloped,allow upside down},draw={rgb:blue,4;red,1;green,1;black,3},opacity=0.7]
\tikzset{connection path xz mid/.style={
  ultra thick,
  draw=\edgecolor,
  opacity=0.7,
  decoration={
    markings,
    mark=at position 0.5 with {\arrow[scale=1.12]{Stealth[length=4.5mm,width=3.4mm,line width=0.8mm,fill={\edgecolor}]}}
  },
  postaction={decorate}
}}
\tikzset{connection path xy mid/.style={
  ultra thick,
  draw=\edgecolor,
  opacity=0.7,
  decoration={
    markings,
    mark=at position 0.5 with {\arrow[scale=1.12]{Stealth[length=4.5mm,width=3.4mm,line width=0.8mm,fill={\edgecolor}]}}
  },
  postaction={decorate}
}}
\tikzset{connection path plain/.style={
  ultra thick,
  draw=\edgecolor,
  opacity=0.7
}}
"""

# layers definition

def to_input(pathfile, to="(-3,0,0)", width=8, height=8, name="temp"):
    """Image on the same ZY billboard as Conv boxes; ``transform shape`` applies the 3D canvas tilt."""

    def _dim_cm(v) -> str:
        if isinstance(v, (int, float)):
            return f"{float(v):.3f}"
        return str(v)

    w = _dim_cm(width)
    h = _dim_cm(height)
    return (
        r"""
\node[canvas is zy plane at x=0, transform shape, inner sep=0pt] ("""
        + name
        + r""") at """
        + to
        + r""" {\includegraphics[width="""
        + w
        + r"""cm,height="""
        + h
        + r"""cm]{"""
        + pathfile
        + r"""}};
"""
    )


# TikZ ``x`` span for ``to_input_frame_strip``: centers are evenly spaced from ``x_left`` to ``x_right``.
# Smaller ``abs(x_right - x_left)`` packs frames closer (same for all ``n``; gap is span / max(n - 1, 1)).
INPUT_FRAME_STRIP_X_LEFT = -3.0
INPUT_FRAME_STRIP_X_RIGHT = -0.75


def to_input_frame_strip(
    rel_paths: list[str],
    *,
    width_cm: float | int,
    height_cm: float | int,
    x_left: float = INPUT_FRAME_STRIP_X_LEFT,
    x_right: float = INPUT_FRAME_STRIP_X_RIGHT,
    name_prefix: str = "inputstrip",
) -> str:
    """Several input frames on the same ZY billboard style as ``to_input``, evenly spaced in ``x``.

    ``rel_paths`` are paths as written in ``.tex`` (e.g. ``input_bundle/000.png``). ``x_left`` / ``x_right``
    are TikZ units; the first path is placed at ``x_left`` (earlier time), the last at ``x_right`` (nearest
    the Video box at the origin). Tune spacing via ``INPUT_FRAME_STRIP_X_LEFT`` / ``INPUT_FRAME_STRIP_X_RIGHT``
    or by passing ``x_left`` / ``x_right`` here.
    """

    if not rel_paths:
        return ""

    def _dim_cm(v: float | int) -> str:
        if isinstance(v, (int, float)):
            return f"{float(v):.3f}"
        return str(v)

    w = _dim_cm(width_cm)
    h = _dim_cm(height_cm)
    n = len(rel_paths)
    chunks: list[str] = []
    for i, pathfile in enumerate(rel_paths):
        if n <= 1:
            x = (x_left + x_right) / 2.0
        else:
            x = x_left + i * (x_right - x_left) / (n - 1)
        xs = f"{x:.4f}".rstrip("0").rstrip(".")
        name = name_prefix + str(i)
        chunks.append(
            r"""
\node[canvas is zy plane at x=0, transform shape, inner sep=0pt] ("""
            + name
            + r""") at ("""
            + xs
            + r""",0,0) {\includegraphics[width="""
            + w
            + r"""cm,height="""
            + h
            + r"""cm]{"""
            + pathfile
            + r"""}};
"""
        )
    return "".join(chunks)


def _box_xlabel_array_one(n_filer: str | int) -> str:
    """One pgf ``array`` slot for ``xlabel`` (comma before closing); use after ``xlabel=`` key.

    Quoted strings must keep ``$...$`` so math subscripts typeset; stripping to inner
    ``C_{\\mathrm{in}}`` lets pgfmath ``array()`` collapse the label to ``C``.
    """

    if isinstance(n_filer, str):
        safe = n_filer.strip().replace('"', "''")
        return '{{"' + safe + '", }}'
    return "{{ " + str(n_filer) + ", }}"


def _box_xlabel_line(n_filer: int | str | None) -> str:
    """Full ``        xlabel=...`` line for ``Box`` / ``RightBandedBox`` (trailing newline)."""

    if n_filer is None:
        return '        xlabel={{"", }},\n'
    return "        xlabel=" + _box_xlabel_array_one(n_filer) + ",\n"


# Conv
def to_Conv(
    name,
    s_filer=256,
    n_filer=64,
    offset="(0,0,0)",
    to="(0,0,0)",
    width=1,
    height=40,
    depth=40,
    caption=" ",
    ylabel=None,
    *,
    fill_tex: str = r"\ConvColor",
    caption_yshift: float | int | None = None,
    caption_xshift: float | int | None = None,
    caption_anchor: str | None = None,
    opacity: float | int | None = None,
):
    yl = "" if ylabel is None else "        ylabel=" + str(ylabel) + ",\n"
    cap_y = (
        "        caption yshift=" + str(caption_yshift) + ",\n" if caption_yshift is not None else ""
    )
    cap_x = (
        "        caption xshift=" + str(caption_xshift) + ",\n" if caption_xshift is not None else ""
    )
    cap_a = _box_caption_anchor_line(caption_anchor)
    op_line = "        opacity=" + str(opacity) + ",\n" if opacity is not None else ""
    xl = _box_xlabel_line(n_filer)
    return (
        r"""
\pic[shift={"""
        + offset
        + r"""}] at """
        + to
        + r"""
    {Box={
        name="""
        + name
        + """,
        caption="""
        + caption
        + """,
"""
        + cap_y
        + cap_x
        + cap_a
        + xl
        + "        zlabel="
        + str(s_filer)
        + ",\n"
        + yl
        + "        fill="
        + fill_tex
        + r""",
"""
        + op_line
        + r"""        height="""
        + str(height)
        + r""",
        width="""
        + str(width)
        + r""",
        depth="""
        + str(depth)
        + r"""
        }
    };
"""
    )

# Fully-connected (VSRM head): narrow in the pipeline axis, tall for a “column” readout metaphor.
def to_Fc(
    name,
    n_in: int,
    offset="(0,0,0)",
    to="(0,0,0)",
    width=2.2,
    height=28,
    depth=4,
    caption=" ",
    zlabel: str | None = None,
    pipeline_xlabel: int | str | None = None,
):
    z_tex = zlabel if zlabel is not None else f"${n_in}$"
    xl = _box_xlabel_line(pipeline_xlabel)
    return (
        r"""
\pic[shift={"""
        + offset
        + r"""}] at """
        + to
        + r"""
    {Box={
        name="""
        + name
        + """,
        caption="""
        + caption
        + """,
"""
        + xl
        + "        zlabel="
        + z_tex
        + r""",
        fill=\FcColor,
        height="""
        + str(height)
        + r""",
        width="""
        + str(width)
        + r""",
        depth="""
        + str(depth)
        + r"""
        }
    };
"""
    )


def _rb_xlabel_pair(n0, n1) -> str:
    """RightBandedBox ``xlabel`` slot: ints unquoted; strings quoted for pgf (symbols, \mbox, etc.)."""

    def fmt(v) -> str:
        if isinstance(v, str):
            return '"' + v.replace('"', "''") + '"'
        return str(v)

    return "{{ " + fmt(n0) + ", " + fmt(n1) + " }}"


def _pgf_braced(s) -> str:
    """Wrap TikZ/pgf key values that contain ``$`` or spaces so ``zlabel=$T$`` does not break the key parser."""
    t = str(s).strip()
    return "{" + t + "}"


_CAPTION_ANCHORS = frozenset({"sw", "ne"})


def _box_caption_anchor_line(anchor: str | None) -> str:
    """``caption anchor`` for ``Box`` / ``RightBandedBox`` (``sw`` = bottom-front, ``ne`` = top-near-east)."""

    if anchor is None:
        return ""
    a = anchor.strip().lower()
    if a not in _CAPTION_ANCHORS:
        raise ValueError("caption anchor must be 'sw', 'ne', or None, got " + repr(anchor))
    if a == "sw":
        return ""
    return "        caption anchor=" + a + ",\n"


def _rb_width_tex(width: tuple[int | float, ...]) -> str:
    return "{ " + " , ".join(str(w) for w in width) + " }"


# Conv,Conv,relu
# Bottleneck
def to_ConvConvRelu( name, s_filer=256, n_filer=(64,64), offset="(0,0,0)", to="(0,0,0)", width=(2,2), height=40, depth=40, caption=" ", zlabel_pos=None, *, caption_yshift=None, caption_xshift=None, caption_anchor=None ):
    if len(width) == 1:
        if isinstance(n_filer, str):
            xl = _box_xlabel_array_one(n_filer)
        elif isinstance(n_filer, tuple) and len(n_filer) == 1:
            xl = _box_xlabel_array_one(n_filer[0])
        else:
            raise TypeError(
                "to_ConvConvRelu with single width expects n_filer str or 1-tuple, got "
                + repr(n_filer)
            )
    elif len(width) == 2:
        xl = _rb_xlabel_pair(n_filer[0], n_filer[1])
    else:
        raise ValueError("to_ConvConvRelu width must be length 1 or 2")

    z_line = ""
    if str(s_filer).strip():
        z_line = "        zlabel=" + _pgf_braced(s_filer) + ",\n"
        if zlabel_pos is not None:
            z_line += "        zlabel pos=" + str(zlabel_pos) + ",\n"

    cap_y = ""
    if caption_yshift is not None:
        cap_y = "        caption yshift=" + str(caption_yshift) + ",\n"
    cap_x = ""
    if caption_xshift is not None:
        cap_x = "        caption xshift=" + str(caption_xshift) + ",\n"
    cap_a = _box_caption_anchor_line(caption_anchor)

    return r"""
\pic[shift={ """+ offset +""" }] at """+ to +""" 
    {RightBandedBox={
        name="""+ name +""",
        caption="""+ caption +""",
""" + cap_y + cap_x + cap_a + """        xlabel=""" + xl + r""",
""" + z_line + r"""        fill=\ConvColor,
        bandfill=\ConvReluColor,
        height="""+ str(height) +""",
        width=""" + _rb_width_tex(width) + r""",
        depth="""+ str(depth) +"""
        }
    };
"""



# Pool
def to_Pool(
    name,
    offset="(0,0,0)",
    to="(0,0,0)",
    width=1,
    height=32,
    depth=32,
    opacity=0.5,
    caption=" ",
    zlabel=None,
    n_filer=None,
):
    extra = ""
    if zlabel is not None:
        extra += "        zlabel=" + str(zlabel) + ",\n"
    if n_filer is not None:
        extra += "        xlabel={{" + str(n_filer) + ", }},\n"
    return r"""
\pic[shift={ """+ offset +""" }] at """+ to +""" 
    {Box={
        name="""+name+""",
        caption="""+ caption +r""",
""" + extra + r"""        fill=\PoolColor,
        opacity="""+ str(opacity) +""",
        height="""+ str(height) +""",
        width="""+ str(width) +""",
        depth="""+ str(depth) +"""
        }
    };
"""

# unpool4, 
def to_UnPool(name, offset="(0,0,0)", to="(0,0,0)", width=1, height=32, depth=32, opacity=0.5, caption=" "):
    return r"""
\pic[shift={ """+ offset +""" }] at """+ to +""" 
    {Box={
        name="""+ name +r""",
        caption="""+ caption +r""",
        fill=\UnpoolColor,
        opacity="""+ str(opacity) +""",
        height="""+ str(height) +""",
        width="""+ str(width) +""",
        depth="""+ str(depth) +"""
        }
    };
"""



def to_ConvRes( name, s_filer=256, n_filer=64, offset="(0,0,0)", to="(0,0,0)", width=6, height=40, depth=40, opacity=0.2, caption=" " ):
    return r"""
\pic[shift={ """+ offset +""" }] at """+ to +""" 
    {RightBandedBox={
        name="""+ name + """,
        caption="""+ caption + """,
        xlabel={{ """+ str(n_filer) + """, }},
        zlabel="""+ str(s_filer) +r""",
        fill={rgb:white,1;black,3},
        bandfill={rgb:white,1;black,2},
        opacity="""+ str(opacity) +""",
        height="""+ str(height) +""",
        width="""+ str(width) +""",
        depth="""+ str(depth) +"""
        }
    };
"""


# ConvSoftMax
def to_ConvSoftMax( name, s_filer=40, offset="(0,0,0)", to="(0,0,0)", width=1, height=40, depth=40, caption=" " ):
    return r"""
\pic[shift={"""+ offset +"""}] at """+ to +""" 
    {Box={
        name=""" + name +""",
        caption="""+ caption +""",
        zlabel="""+ str(s_filer) +""",
        fill=\SoftmaxColor,
        height="""+ str(height) +""",
        width="""+ str(width) +""",
        depth="""+ str(depth) +"""
        }
    };
"""

# SoftMax
def to_SoftMax( name, s_filer=10, offset="(0,0,0)", to="(0,0,0)", width=1.5, height=3, depth=25, opacity=0.8, caption=" " ):
    return r"""
\pic[shift={"""+ offset +"""}] at """+ to +""" 
    {Box={
        name=""" + name +""",
        caption="""+ caption +""",
        xlabel={{" ","dummy"}},
        zlabel="""+ str(s_filer) +""",
        fill=\SoftmaxColor,
        opacity="""+ str(opacity) +""",
        height="""+ str(height) +""",
        width="""+ str(width) +""",
        depth="""+ str(depth) +"""
        }
    };
"""

def to_Sum( name, offset="(0,0,0)", to="(0,0,0)", radius=2.5, opacity=0.6):
    return r"""
\pic[shift={"""+ offset +"""}] at """+ to +""" 
    {Ball={
        name=""" + name +""",
        fill=\SumColor,
        opacity="""+ str(opacity) +""",
        radius="""+ str(radius) +""",
        logo=$+$
        }
    };
"""


def to_res_sum_ball(
    name: str,
    east_of: str,
    west_of: str,
    *,
    pos: float = 0.5,
    radius: float = 2.5,
    opacity: float = 0.6,
) -> str:
    """Residual add node: ``+`` ball along the feedforward gap (uses TikZ ``calc``; see ``init.tex``).

    Default ``opacity=1`` so the shaded sphere fully occludes paths on layers behind it.
    """

    at = rf"($ ({east_of})!{pos}!({west_of}) $)"
    return to_Sum(name, offset="(0,0,0)", to=at, radius=radius, opacity=opacity)


def to_tensor_io_box(
    name: str,
    caption: str = " ",
    offset: str = "(0,0,0)",
    to: str = "(0,0,0)",
    width: float = 1.5,
    height: float = 1.5,
    depth: int = 48,
    *,
    xlabel: str = r"$C$",
    zlabel: str = r"$\leftarrow T$",
    zlabel_pos: float = 1,
    opacity: float = 0.58,
    caption_yshift: float | None = None,
    caption_xshift: float | None = None,
    caption_anchor: str | None = None,
) -> str:
    """Solid ``Box`` (no conv band): tensor I/O slab (e.g.\ TCN I/O, ResBlock I/O); neutral fill vs conv stack."""

    xl = _box_xlabel_array_one(xlabel)
    z_line = "        zlabel=" + _pgf_braced(zlabel) + ",\n"
    zpos_line = "        zlabel pos=" + str(zlabel_pos) + ",\n"
    cap_y = (
        "        caption yshift=" + str(caption_yshift) + ",\n" if caption_yshift is not None else ""
    )
    cap_x = (
        "        caption xshift=" + str(caption_xshift) + ",\n" if caption_xshift is not None else ""
    )
    cap_a = _box_caption_anchor_line(caption_anchor)
    return (
        r"""
\pic[shift={ """
        + offset
        + r""" }] at """
        + to
        + r"""
    {Box={
        name="""
        + name
        + """,
        caption="""
        + caption
        + """,
"""
        + cap_y
        + cap_x
        + cap_a
        + """        xlabel="""
        + xl
        + """,
"""
        + z_line
        + zpos_line
        + r"""        fill=\TensorIoColor,
        opacity="""
        + str(opacity)
        + r""",
        height="""
        + str(height)
        + r""",
        width="""
        + str(width)
        + r""",
        depth="""
        + str(depth)
        + r"""
        }
    };
"""
    )


def to_connection( of, to):
    return r"""
\draw [connection]  ("""+of+"""-east)    -- node {\midarrow} ("""+to+"""-west);
"""


def to_connection_through_sum(left_block: str, sum_ball: str, right_block: str) -> str:
    """Feedforward split at a residual ``Ball``: left slab → ``sum``-west, ``sum``-east → right slab."""

    return (
        r"""
\draw [connection]  ("""
        + left_block
        + r"""-east)    -- node {\midarrow} ("""
        + sum_ball
        + r"""-west);
\draw [connection]  ("""
        + sum_ball
        + r"""-east)    -- node {\midarrow} ("""
        + right_block
        + r"""-west);
"""
    )


def to_res_skip_manhattan_xz_to_sum(
    sum_ball: str,
    in_east: str,
    first_conv_west: str,
    *,
    step_z: float = 7.0,
    coord_prefix: str = "tcn_skip_in_res",
    z_first_leg_sign: int = -1,
    origin: str | None = None,
    sum_join: str = "near",
) -> str:
    """Manhattan skip in the ``xz`` plane (at ``y=0``): first leg along canvas ``z``, across in ``x``, then ``connection`` to the sum ball.

    If ``origin`` is set (e.g. ``"tcn_res_12-anchor"``, ``"tcn_res_12-far"``), the skip starts at that coordinate.
    Otherwise ``in_east`` / ``first_conv_west`` define a segment and the skip starts at its midpoint.

    ``sum_join`` is ``"near"`` or ``"far"``: suffix for the destination ball rim (``Ball`` ``-near`` / ``-far``).

    ``z_first_leg_sign`` is ``-1`` (default) or ``+1``: first leg is ``++(0, sign * step_z)`` in the xz canvas
    (flip for alternating depth direction on screen).

    Uses ``canvas is xz plane at y=0`` so the horizontal leg stays clear of depth-edge ``T`` labels.
    In-plane legs use ``connection path xz mid`` (markings: Stealth follows path tangent under the xz canvas).
    The final segment uses ``[connection]`` with ``node {\\midarrow}`` like other 3D edges; it ends at ``sum_ball-{sum_join}``.
    Wrapped in ``on background layer`` so the path composites behind the sum Ball (see ``backgrounds`` in ``init.tex``).
    """

    if z_first_leg_sign not in (-1, 1):
        raise ValueError("z_first_leg_sign must be -1 or 1")
    if sum_join not in ("near", "far"):
        raise ValueError('sum_join must be "near" or "far"')

    c0, c1, c2 = f"{coord_prefix}0", f"{coord_prefix}1", f"{coord_prefix}2"
    dz = _tikz_float_str(z_first_leg_sign * float(step_z))
    if origin is not None:
        c0_at = r"\path coordinate (" + c0 + r") at (" + origin + r");"
    else:
        c0_at = (
            r"\path coordinate ("
            + c0
            + r""") at ($ ("""
            + in_east
            + ")!0.5!("
            + first_conv_west
            + r""") $);"""
        )
    return (
        c0_at
        + r"""
\begin{scope}[canvas is xz plane at y=0]
\draw [connection path xz mid]  ("""
        + c0
        + r""")    -- ++(0,"""
        + dz
        + r""") coordinate ("""
        + c1
        + r""");
\path coordinate ("""
        + c2
        + r""") at ("""
        + c1
        + r""" -| """
        + sum_ball
        + r"""-anchor);
\draw [connection path xz mid]  ("""
        + c1
        + r""")    -- ("""
        + c2
        + r""");
\end{scope}
\draw [connection]  ("""
        + c2
        + r""")    -- node {\midarrow} ("""
        + sum_ball
        + "-"
        + sum_join
        + r""");
"""
    )


def to_res_skip_manhattan_xy_to_sum(
    sum_ball: str,
    in_east: str,
    first_conv_west: str,
    *,
    step_y: float = 7.0,
    coord_prefix: str = "tcn_skip_xy_res",
    y_first_leg_sign: int = 1,
    origin: str | None = None,
    sum_join: str = "near",
) -> str:
    """Manhattan skip in the ``xy`` plane (at ``z=0``): first leg along canvas ``y`` (vertical on screen), across in ``x``, then ``connection`` to the sum ball.

    Use for ResBlock-style figures where a path *above* the pipeline (xy plateau) reads better than the TCN xz routing.
    TCN diagrams keep ``to_res_skip_manhattan_xz_to_sum`` (xz) to stay clear of depth/time edge labels.

    ``y_first_leg_sign`` is ``+1`` (default) or ``-1``: first leg is ``++(0, sign * step_y)`` in the xy canvas.

    ``sum_join`` names a ``Ball`` rim anchor (see ``Ball.sty``): ``near`` / ``far`` (depth), ``north`` / ``south`` / ``east`` / ``west``, or ``anchor``.
    For a skip that drops onto the top of the ball, use ``north``.

    Unlike the xz-plane helper, this path is **not** wrapped in ``on background layer`` so it paints after prior
    ``\\pic`` nodes and reads above boxes in **screen order** (ResBlock skip over the input slab).
    """

    if y_first_leg_sign not in (-1, 1):
        raise ValueError("y_first_leg_sign must be -1 or 1")
    _xy_sum_joins = frozenset({"near", "far", "north", "south", "east", "west", "anchor"})
    if sum_join not in _xy_sum_joins:
        raise ValueError(f'sum_join must be one of {sorted(_xy_sum_joins)!r}, got {sum_join!r}')

    c0, c1, c2 = f"{coord_prefix}0", f"{coord_prefix}1", f"{coord_prefix}2"
    dy = _tikz_float_str(y_first_leg_sign * float(step_y))
    if origin is not None:
        c0_at = r"\path coordinate (" + c0 + r") at (" + origin + r");"
    else:
        c0_at = (
            r"\path coordinate ("
            + c0
            + r""") at ($ ("""
            + in_east
            + ")!0.5!("
            + first_conv_west
            + r""") $);"""
        )
    return (
        c0_at
        + r"""
\begin{scope}[canvas is xy plane at z=0]
\draw [connection path xy mid]  ("""
        + c0
        + r""")    -- ++(0,"""
        + dy
        + r""") coordinate ("""
        + c1
        + r""");
\path coordinate ("""
        + c2
        + r""") at ("""
        + c1
        + r""" -| """
        + sum_ball
        + "-"
        + sum_join
        + r""");
\draw [connection path xy mid]  ("""
        + c1
        + r""")    -- ("""
        + c2
        + r""");
\end{scope}
\draw [connection]  ("""
        + c2
        + r""")    -- node {\midarrow} ("""
        + sum_ball
        + "-"
        + sum_join
        + r""");
"""
    )


def to_res_skip_manhattan_xy_to_sum_with_proj(
    sum_ball: str,
    in_east: str,
    first_conv_west: str,
    *,
    step_y: float = 7.0,
    coord_prefix: str = "tcn_skip_xy_res",
    y_first_leg_sign: int = 1,
    origin: str | None = None,
    sum_join: str = "near",
    proj_name: str = "rb_skip_proj",
    proj_w_coord: str | None = None,
    proj_half_width_x: float = 0.2,
    proj_caption: str = " ",
    proj_width: tuple[int | float, ...] | int | float = (2,),
    proj_height: int = 18,
    proj_depth: int = 18,
    proj_n_filer: str | tuple[str, ...] = (r"$C_{\mathrm{out}}$",),
    proj_zlabel: str = "",
    proj_caption_yshift: int | None = None,
    proj_caption_xshift: int | None = None,
    proj_caption_anchor: str | None = None,
) -> str:
    """Same Manhattan ``xy`` skip as ``to_res_skip_manhattan_xy_to_sum``, with a 1×1×1 proj slab on the plateau.

    The horizontal leg still runs from ``coord_prefix+1`` to ``coord_prefix+2`` at ``(c1 -| sum_ball-sum_join)``;
    ``coord_prefix+2`` matches the no-proj helper. The proj ``\\pic`` west is at the midpoint of that segment minus
    ``proj_half_width_x`` along world $+x$ (``Box`` ``scale=0.2``). Plain ``Box``: skip projection is conv-only (no activation on that branch).
    Optional ``proj_caption_anchor="ne"`` places the skip caption at the top-near-east corner (see ``caption anchor`` in ``Box.sty``).
    """

    if isinstance(proj_width, (int, float)):
        proj_width = (float(proj_width),)
    elif not isinstance(proj_width, tuple):
        raise TypeError(
            "proj_width must be a number or tuple of segment widths, got "
            + type(proj_width).__name__
        )

    if y_first_leg_sign not in (-1, 1):
        raise ValueError("y_first_leg_sign must be -1 or 1")
    _xy_sum_joins = frozenset({"near", "far", "north", "south", "east", "west", "anchor"})
    if sum_join not in _xy_sum_joins:
        raise ValueError(f'sum_join must be one of {sorted(_xy_sum_joins)!r}, got {sum_join!r}')

    c0, c1, c2 = f"{coord_prefix}0", f"{coord_prefix}1", f"{coord_prefix}2"
    pw = proj_w_coord if proj_w_coord is not None else f"{coord_prefix}_proj_w"
    dy = _tikz_float_str(y_first_leg_sign * float(step_y))
    hwx = _tikz_float_str(float(proj_half_width_x))
    if origin is not None:
        c0_at = r"\path coordinate (" + c0 + r") at (" + origin + r");"
    else:
        c0_at = (
            r"\path coordinate ("
            + c0
            + r""") at ($ ("""
            + in_east
            + ")!0.5!("
            + first_conv_west
            + r""") $);"""
        )
    sum_anchor = sum_ball + "-" + sum_join
    z_line = ""
    if str(proj_zlabel).strip():
        z_line = "        zlabel=" + _pgf_braced(proj_zlabel) + ",\n"

    cap_y = ""
    if proj_caption_yshift is not None:
        cap_y = "        caption yshift=" + str(proj_caption_yshift) + ",\n"
    cap_x = ""
    if proj_caption_xshift is not None:
        cap_x = "        caption xshift=" + str(proj_caption_xshift) + ",\n"
    cap_a = _box_caption_anchor_line(proj_caption_anchor)

    if len(proj_width) == 1:
        if isinstance(proj_n_filer, str):
            xl = _box_xlabel_array_one(proj_n_filer)
        elif isinstance(proj_n_filer, tuple) and len(proj_n_filer) == 1:
            xl = _box_xlabel_array_one(proj_n_filer[0])
        else:
            raise TypeError(
                "proj_n_filer must be str or 1-tuple for single-segment proj width, got "
                + repr(proj_n_filer)
            )
    else:
        raise ValueError("to_res_skip_manhattan_xy_to_sum_with_proj expects proj_width of length 1")

    lead = (
        c0_at
        + r"""
\begin{scope}[canvas is xy plane at z=0]
\draw [connection path xy mid]  ("""
        + c0
        + r""")    -- ++(0,"""
        + dy
        + r""") coordinate ("""
        + c1
        + r""");
\path coordinate ("""
        + c2
        + r""") at ("""
        + c1
        + r""" -| """
        + sum_anchor
        + r""");
\end{scope}
\path coordinate ("""
        + pw
        + r""") at ($ ("""
        + c1
        + r""")!0.5!("""
        + c2
        + r""") + (-"""
        + hwx
        + r""",0,0) $);
"""
    )
    proj_w0 = proj_width[0]
    proj_pic = (
        r"""
\pic[shift={(0,0,0)}] at ("""
        + pw
        + r""")
    {Box={
        name="""
        + proj_name
        + r""",
        caption="""
        + proj_caption
        + r""",
"""
        + cap_y
        + cap_x
        + cap_a
        + "        xlabel="
        + xl
        + r""",
"""
        + z_line
        + r"""        fill=\ConvColor,
        height="""
        + str(proj_height)
        + r""",
        width="""
        + str(proj_w0)
        + r""",
        depth="""
        + str(proj_depth)
        + r"""
        }
    };
"""
    )
    tail = (
        r"""
\begin{scope}[canvas is xy plane at z=0]
\draw [connection path xy mid]  ("""
        + c1
        + r""")    -- ("""
        + proj_name
        + r"""-west);
\draw [connection path xy mid]  ("""
        + proj_name
        + r"""-east)    -- ("""
        + c2
        + r""");
\end{scope}
\draw [connection]  ("""
        + c2
        + r""")    -- node {\midarrow} ("""
        + sum_anchor
        + r""");
"""
    )
    return lead + proj_pic + tail


def _tikz_float_str(x: float) -> str:
    s = f"{x:.12f}".rstrip("0").rstrip(".")
    return s if s else "0"


def to_slab_depth_time_ticks(block: str, num_steps: int) -> str:
    """Short segments *along* the time edge (not in ``y``), so they do not read as vertical spikes."""

    if num_steps < 1:
        return ""
    lines: list[str] = []
    for k in range(num_steps + 1):
        fa = max(0.0, (k - 0.5) / num_steps)
        fb = min(1.0, (k + 0.5) / num_steps)
        lo = "0" if fa <= 0.0 else _tikz_float_str(fa)
        hi = "1" if fb >= 1.0 else _tikz_float_str(fb)
        for a, b in (("farwest", "nearwest"), ("fareast", "neareast")):
            lines.append(
                rf"\draw[very thin,gray!55,opacity=0.85] ($ ({block}-{a})!{lo}!({block}-{b}) $) "
                rf"-- ($ ({block}-{a})!{hi}!({block}-{b}) $);"
            )
    return "\n".join(lines) + "\n"


def _subsample_int_range(lo: int, hi: int, max_n: int | None) -> list[int]:
    """Inclusive ``lo..hi``; if ``max_n`` is set and smaller, pick evenly spaced integers (deduped)."""

    ts = list(range(lo, hi + 1))
    if not ts:
        return []
    if max_n is None or max_n >= len(ts):
        return ts
    ng = max(1, int(max_n))
    if ng >= len(ts):
        return ts
    if ng == 1:
        return [ts[len(ts) // 2]]
    out: list[int] = []
    for j in range(ng):
        idx = int(round(j * (len(ts) - 1) / (ng - 1)))
        out.append(ts[idx])
    seen: set[int] = set()
    dedup: list[int] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            dedup.append(x)
    return dedup


def to_causal_temporal_tap_arrows(
    from_block: str,
    to_block: str,
    num_steps: int,
    num_groups: int | None = None,
    dilation: int = 1,
    emphasize: str = "none",
    muted_opacity: float = 0.38,
    muted_dash: str = "densely dashed",
    emphasize_ts: tuple[int, ...] | None = None,
    emphasize_front_ranks: tuple[int, ...] | None = None,
) -> str:
    """Repeat three-arrow groups along time: ``fareast→neareast`` on *from* east, ``farwest→nearwest`` on *to* west.

    Partway modifiers require **numeric** fractions; ``\foreach`` + ``\pgfmathsetmacro`` macros are not
    reliably expanded inside ``!(...)!``, so this emits explicit ``\\draw`` lines from Python.

    Causal kernel size 3, dilation ``d``: for each output time index ``t`` (fraction ``t/num_steps`` on *to*),
    draw from ``(t-2d)/t-d/t`` on *from* only when the source index is **≥ 0** (omit legs before the start of
    the sequence). ``t`` runs ``2 .. num_steps`` so outputs cover the full time edge; optional ``num_groups``
    subsamples which ``t`` values are drawn.

    Emphasis (first match wins):
      - ``emphasize_ts``: explicit output time indices ``t`` (same units as ``2 .. num_steps``) to draw solid;
        must appear in the subsampled ``ts`` list. Use this to highlight fixed positions without changing
        ``num_groups``.
      - ``emphasize_front_ranks``: 1-based ranks from **near the viewer** (largest ``t``) toward **far**
        (smaller ``t``), relative to the current ``ts`` length. Subsampling to fewer groups shifts which ``t``
        each rank maps to—prefer ``emphasize_ts`` when you want stable highlights with full resolution.
      - ``emphasize == "max_t"``: only the largest ``t`` in ``ts`` is solid.
      - otherwise uniform solid style.

    Wrapped in ``on background layer`` so conv slabs drawn earlier stay on top (see ``init.tex``).
    """

    if num_steps < 2:
        return ""
    d = max(1, int(dilation))
    fb = from_block
    tb = to_block
    head = r"[-{Stealth[length=1.2mm,width=0.9mm]},semithick"
    sty_primary = head + r",draw=\edgecolor,opacity=0.72]"
    mo = _tikz_float_str(float(muted_opacity))
    dash_seg = ("," + muted_dash.strip()) if muted_dash.strip() else ""
    sty_muted = head + dash_seg + rf",draw=\edgecolor,opacity={mo}]"
    lines: list[str] = []

    def pf(n: int) -> str:
        return _tikz_float_str(n / num_steps)

    t_lo, t_hi = 2, num_steps
    if t_lo > t_hi:
        return ""
    ts = _subsample_int_range(t_lo, t_hi, num_groups)
    primary_ts: set[int] | None = None
    if emphasize_ts is not None and ts:
        want = {int(x) for x in emphasize_ts}
        primary_ts = {t for t in ts if t in want}
    elif emphasize_front_ranks is not None and ts:
        n = len(ts)
        primary_ts = {ts[n - k] for k in emphasize_front_ranks if isinstance(k, int) and 1 <= k <= n}
    elif emphasize == "max_t" and ts:
        primary_ts = {max(ts)}

    for t in ts:
        if primary_ts is None:
            sty = sty_primary
        else:
            sty = sty_primary if t in primary_ts else sty_muted
        p_out = pf(t)
        for sk in (t - 2 * d, t - d, t):
            if sk < 0:
                continue
            ps = pf(sk)
            lines.append(
                rf"\draw{sty} ($ ({fb}-fareast)!{ps}!({fb}-neareast) $) "
                rf"-- ($ ({tb}-farwest)!{p_out}!({tb}-nearwest) $);"
            )
    if not lines:
        return ""
    body = "\n".join(lines) + "\n"
    return (
        r"""
\begin{scope}[on background layer]
"""
        + body
        + r"""\end{scope}
"""
    )




def to_adjacent_slabs_span_label_below(
    left_block: str,
    right_block: str,
    label: str = r"\textbf{TCN Block 1}",
    below_front_pt: float = 65.0,
    *,
    xshift_pt: float = 0.0,
) -> str:
    """Below both layer captions: anchor on the *near* bottom seam (same face as RightBandedBox captions).

    ``*-south`` is at ``z=0`` while captions hang from the near-bottom center minus 30pt, so the
    ``south`` midpoint projected too high. Use ``nearsoutheast``/``nearsouthwest`` and a larger drop.
    """

    d = str(int(below_front_pt)) if below_front_pt == int(below_front_pt) else str(below_front_pt)
    xs = _tikz_float_str(xshift_pt)
    return (
        r"\path ($ ("
        + left_block
        + r"-nearsoutheast)!0.5!("
        + right_block
        + r"-nearsouthwest) $) +("
        + xs
        + r"pt,-"
        + d
        + r"pt) node[anchor=north, inner sep=2pt, align=center] {"
        + label
        + r"};"
        + "\n"
    )


def to_connection_curved(of, to, out_angle: float = 16.0, in_angle: float = 164.0):
    """Gentle bend between blocks (e.g. wide FC into narrow logits column)."""
    return (
        r"\draw [connection] ("
        + of
        + r"-east) to[out="
        + str(out_angle)
        + r",in="
        + str(in_angle)
        + r"] node[pos=0.5] {\midarrow} ("
        + to
        + r"-west);"
        + "\n"
    )

def to_skip( of, to, pos=1.25):
    return r"""
\path ("""+ of +"""-southeast) -- ("""+ of +"""-northeast) coordinate[pos="""+ str(pos) +"""] ("""+ of +"""-top) ;
\path ("""+ to +"""-south)  -- ("""+ to +"""-north)  coordinate[pos="""+ str(pos) +"""] ("""+ to +"""-top) ;
\draw [copyconnection]  ("""+of+"""-northeast)  
-- node {\copymidarrow}("""+of+"""-top)
-- node {\copymidarrow}("""+to+"""-top)
-- node {\copymidarrow} ("""+to+"""-north);
"""


def to_copyconnection_rect(from_anchor: str, to_anchor: str) -> str:
    """Residual-style path: horizontal leg along from_anchor's y, then vertical to to_anchor (orthogonal, no diagonal)."""
    return (
        r"\draw [copyconnection] ("
        + from_anchor
        + r") -- node {\copymidarrow} ("
        + from_anchor
        + r" -| "
        + to_anchor
        + r") -- node {\copymidarrow} ("
        + to_anchor
        + r");"
        + "\n"
    )


def to_copyconnection_up_right_down(from_anchor: str, to_anchor: str, lift_pt: float = 40.0) -> str:
    """Up from source, horizontal toward target, then down into target (orthogonal U, polished skip style)."""
    return (
        r"\coordinate (vsrm-skip-up) at ([yshift="
        + str(lift_pt)
        + r"pt]"
        + from_anchor
        + r");\coordinate (vsrm-skip-mid) at (vsrm-skip-up -| "
        + to_anchor
        + r");\draw [copyconnection] ("
        + from_anchor
        + r") -- node {\copymidarrow} (vsrm-skip-up) -- node {\copymidarrow} (vsrm-skip-mid) -- node {\copymidarrow} ("
        + to_anchor
        + r");"
        + "\n"
    )

def to_end():
    return r"""
\end{tikzpicture}
\end{document}
"""


def to_generate( arch, pathname="file.tex" ):
    with open(pathname, "w") as f: 
        for c in arch:
            print(c)
            f.write( c )

# ---------------------------------------------------------------------------
# Backward-compatible aliases (historic ``to_tcn_*`` / ``_tcn_*`` names).
# Prefer the diagram-neutral names above; emitted TikZ is identical.
# ---------------------------------------------------------------------------
to_tcn_res_sum_ball = to_res_sum_ball
to_tcn_input_box = to_tensor_io_box
to_tcn_connection_through_sum = to_connection_through_sum
to_tcn_res_skip_manhattan_input_to_sum = to_res_skip_manhattan_xz_to_sum
to_tcn_res_skip_manhattan_xy_input_to_sum = to_res_skip_manhattan_xy_to_sum
to_tcn_res_skip_manhattan_xy_input_to_sum_with_proj = to_res_skip_manhattan_xy_to_sum_with_proj
to_tcn_time_ticks = to_slab_depth_time_ticks
to_tcn_tap_arrow_groups = to_causal_temporal_tap_arrows
to_tcn_super_label = to_adjacent_slabs_span_label_below
_tcn_float_str = _tikz_float_str
_tcn_subsample_int_range = _subsample_int_range

