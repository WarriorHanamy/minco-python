"""
Pybind11 bindings for the polynomial trajectory representation
"""
from __future__ import annotations
import collections.abc
import numpy
import numpy.typing
import typing
__all__: list[str] = ['Piece5', 'Piece7', 'Trajectory5', 'Trajectory7']
class Piece5:
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor.
        """
    @typing.overload
    def __init__(self, duration: typing.SupportsFloat, coeff_mat: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 6]"]) -> None:
        """
        Construct a piece with a duration and a coefficient matrix.
        """
    def __repr__(self) -> str:
        ...
    def check_max_acc_rate(self, max_acc_rate: typing.SupportsFloat) -> bool:
        """
        Check if the acceleration magnitude is always within a certain limit.
        """
    def check_max_vel_rate(self, max_vel_rate: typing.SupportsFloat) -> bool:
        """
        Check if the velocity magnitude is always within a certain limit.
        """
    def get_acc(self, t: typing.SupportsFloat) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]:
        """
        Get acceleration at time t.
        """
    def get_coeff_mat(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 6]"]:
        """
        Get the coefficient matrix (3x(D+1) NumPy array).
        """
    def get_jer(self, t: typing.SupportsFloat) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]:
        """
        Get jerk at time t.
        """
    def get_max_acc_rate(self) -> float:
        """
        Get the maximum acceleration magnitude in this piece.
        """
    def get_max_vel_rate(self) -> float:
        """
        Get the maximum velocity magnitude in this piece.
        """
    def get_pos(self, t: typing.SupportsFloat) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]:
        """
        Get position at time t.
        """
    def get_vel(self, t: typing.SupportsFloat) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]:
        """
        Get velocity at time t.
        """
    @property
    def degree(self) -> int:
        """
        Get the polynomial degree of the piece.
        """
    @property
    def dim(self) -> int:
        """
        Get the spatial dimension (always 3).
        """
    @property
    def duration(self) -> float:
        """
        Get the duration of the piece.
        """
class Piece7:
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor.
        """
    @typing.overload
    def __init__(self, duration: typing.SupportsFloat, coeff_mat: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 8]"]) -> None:
        """
        Construct a piece with a duration and a coefficient matrix.
        """
    def __repr__(self) -> str:
        ...
    def check_max_acc_rate(self, max_acc_rate: typing.SupportsFloat) -> bool:
        """
        Check if the acceleration magnitude is always within a certain limit.
        """
    def check_max_vel_rate(self, max_vel_rate: typing.SupportsFloat) -> bool:
        """
        Check if the velocity magnitude is always within a certain limit.
        """
    def get_acc(self, t: typing.SupportsFloat) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]:
        """
        Get acceleration at time t.
        """
    def get_coeff_mat(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 8]"]:
        """
        Get the coefficient matrix (3x(D+1) NumPy array).
        """
    def get_jer(self, t: typing.SupportsFloat) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]:
        """
        Get jerk at time t.
        """
    def get_max_acc_rate(self) -> float:
        """
        Get the maximum acceleration magnitude in this piece.
        """
    def get_max_vel_rate(self) -> float:
        """
        Get the maximum velocity magnitude in this piece.
        """
    def get_pos(self, t: typing.SupportsFloat) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]:
        """
        Get position at time t.
        """
    def get_vel(self, t: typing.SupportsFloat) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]:
        """
        Get velocity at time t.
        """
    @property
    def degree(self) -> int:
        """
        Get the polynomial degree of the piece.
        """
    @property
    def dim(self) -> int:
        """
        Get the spatial dimension (always 3).
        """
    @property
    def duration(self) -> float:
        """
        Get the duration of the piece.
        """
class Trajectory5:
    def __getitem__(self, arg0: typing.SupportsInt) -> Piece5:
        ...
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor.
        """
    @typing.overload
    def __init__(self, durations: collections.abc.Sequence[typing.SupportsFloat], coeff_mats: collections.abc.Sequence[typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 6]"]]) -> None:
        """
        Construct a trajectory from a list of durations and a list of coefficient matrices.
        """
    def __iter__(self) -> collections.abc.Iterator[Piece5]:
        ...
    def __len__(self) -> int:
        ...
    def __repr__(self) -> str:
        ...
    def append_piece(self, piece: Piece5) -> None:
        """
        Append a piece object to the trajectory.
        """
    def clear(self) -> None:
        """
        Clear all pieces from the trajectory.
        """
    def get_acc(self, t: typing.SupportsFloat) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]:
        """
        Get acceleration at time t along the trajectory.
        """
    def get_jer(self, t: typing.SupportsFloat) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]:
        """
        Get jerk at time t along the trajectory.
        """
    def get_max_acc_rate(self) -> float:
        """
        Get the maximum acceleration magnitude along the entire trajectory.
        """
    def get_max_vel_rate(self) -> float:
        """
        Get the maximum velocity magnitude along the entire trajectory.
        """
    def get_piece_num(self) -> int:
        """
        Get the number of pieces in the trajectory.
        """
    def get_pos(self, t: typing.SupportsFloat) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]:
        """
        Get position at time t along the trajectory.
        """
    def get_vel(self, t: typing.SupportsFloat) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]:
        """
        Get velocity at time t along the trajectory.
        """
    @property
    def durations(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Get a NumPy array of all piece durations.
        """
    @property
    def positions(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, n]"]:
        """
        Get a NumPy array of all waypoint positions.
        """
    @property
    def total_duration(self) -> float:
        """
        Get the total duration of the trajectory.
        """
class Trajectory7:
    def __getitem__(self, arg0: typing.SupportsInt) -> Piece7:
        ...
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor.
        """
    @typing.overload
    def __init__(self, durations: collections.abc.Sequence[typing.SupportsFloat], coeff_mats: collections.abc.Sequence[typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 8]"]]) -> None:
        """
        Construct a trajectory from a list of durations and a list of coefficient matrices.
        """
    def __iter__(self) -> collections.abc.Iterator[Piece7]:
        ...
    def __len__(self) -> int:
        ...
    def __repr__(self) -> str:
        ...
    def append_piece(self, piece: Piece7) -> None:
        """
        Append a piece object to the trajectory.
        """
    def clear(self) -> None:
        """
        Clear all pieces from the trajectory.
        """
    def get_acc(self, t: typing.SupportsFloat) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]:
        """
        Get acceleration at time t along the trajectory.
        """
    def get_jer(self, t: typing.SupportsFloat) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]:
        """
        Get jerk at time t along the trajectory.
        """
    def get_max_acc_rate(self) -> float:
        """
        Get the maximum acceleration magnitude along the entire trajectory.
        """
    def get_max_vel_rate(self) -> float:
        """
        Get the maximum velocity magnitude along the entire trajectory.
        """
    def get_piece_num(self) -> int:
        """
        Get the number of pieces in the trajectory.
        """
    def get_pos(self, t: typing.SupportsFloat) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]:
        """
        Get position at time t along the trajectory.
        """
    def get_vel(self, t: typing.SupportsFloat) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]:
        """
        Get velocity at time t along the trajectory.
        """
    @property
    def durations(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Get a NumPy array of all piece durations.
        """
    @property
    def positions(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, n]"]:
        """
        Get a NumPy array of all waypoint positions.
        """
    @property
    def total_duration(self) -> float:
        """
        Get the total duration of the trajectory.
        """
