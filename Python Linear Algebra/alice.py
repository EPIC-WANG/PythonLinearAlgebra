ABOUT_CONTENT = """## This code implements part of the algorithm in Gilbert Strang's Introduction to Linear Algebra, 
and it is completely written in Python. 

# Copyright (C) 2020  Weizheng Wang

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

COPYRIGHT_CONTENT = """
# Angelina  Copyright (C) 2020  Weizheng Wang
    This program comes with ABSOLUTELY NO WARRANTY; for details, please read LICENSE.txt.
    The copyright of this software is limited by the GNU GENERAL PUBLIC LICENSE.
"""

print('loading, please wait.....', '\b' * 30, end='')
print(flush=True, end='')

import copy
import random
import sys
from typing import Union, Any
from math import *

"""load constants:"""
pi = 3.141592654
e = 2.718281828
j = complex(0, 1)
J = j


def round_all(element: int or float or complex):
    if type(element) is not complex:
        return round(element, 10)
    return complex(round(element.real, 10), round(element.imag, 10))


class AllColumnZeroException(Exception):
    def __init__(self) -> None:
        """
        this error message is used for returning the exception when all the elements in the column are zero (or 0+0j)
        """
        pass


class LengthError(Exception):
    def __int__(self, *args):
        self.args = args


"""DONE"""


class matrix:

    def __init__(self, matrix_: Union[list, None]):
        if type(matrix_[0]) is not list:
            raise SyntaxError('syntax of the matrix is illegal.')
        self.matrix: Union[list, None] = matrix_

    def __add__(self, other):
        return matrix(self.__add(self.matrix, other.matrix))

    def __mul__(self, other):
        return matrix(self.__multiply(self.matrix, other.matrix))

    def __sub__(self, other):
        return matrix(self.__subtract(self.matrix, other.matrix))

    def __abs__(self):
        return matrix(self.__matrix_oper_forloop(r'abs(element)'))

    def __neg__(self):
        return matrix(self.__matrix_oper_forloop(r'-element'))

    def __eq__(self, obj):
        if len(self.matrix) != len(obj.matrix) or len(self.matrix[0]) != len(obj.matrix[0]):
            return False
        try:
            for row in range(len(self.matrix)):
                for column in range(len(self.matrix[0])):
                    assert self.matrix[row][column] == obj.matrix[row][column]
        except AssertionError:
            return False
        else:
            return True

    def __repr__(self):
        if self.matrix:
            return_str = 'matrix:\n'
            for x in self.matrix:
                return_str = return_str + str([round_all(element) for element in x]) + '\n'
            return return_str
        return 'matrix:\n NONE MATRIX\n'

    def __getitem__(self, item):  # make it iterable. 
        raise TypeError("object does not support indexing, use methods: get_row(), get_column()"
                        ", get_matrix_value() instead.")

    def __index__(self):
        raise TypeError("object does not support indexing, use methods: get_row(), get_column()"
                        ", get_matrix_value() instead.")

    def __len__(self):
        raise Exception("object has no length, use .get_size() instead. ")

    def __matrix_oper_forloop(self, eval_operation_for_element: str):
        return_matrix = []
        for row in self.matrix:
            row_matrix = []
            for element in row:
                row_matrix.append(eval(eval_operation_for_element))
            return_matrix.append(row_matrix)
        return return_matrix

    """general mathods below"""

    def get_value(self) -> list:
        return self.matrix

    def get_row(self, row: int):
        return matrix(copy.deepcopy(self.matrix[row]))

    @staticmethod
    def __get_row(my_matrix: list, row: int) -> list:
        return copy.deepcopy(my_matrix[row])

    def get_column(self, column: int):
        return vector([row[column] for row in self.matrix])

    @staticmethod
    def __get_column(my_matrix: list, column: int) -> list:
        return [row[column] for row in my_matrix]

    def get_size(self):
        return len(self.matrix), len(self.matrix[0])

    """
    basic operations:
    all the operations WILL NOT modify the original matrix.
    """

    @staticmethod
    def __multiply(my_matrix: list, target: list) -> list:
        if len(my_matrix[0]) == len(target):
            new_matrix: list = [[0.0] * len(target[0]) for i in range(len(my_matrix))]  # span a empty matrix
            temp_range: range = range(len(target[0]))
            temp_range2: range = range(len(target))
            for i_ in range(len(my_matrix)):
                for j_ in temp_range:
                    new_matrix[i_][j_] = sum([my_matrix[i_][k] * target[k][j_] for k in temp_range2])
            return new_matrix
        raise Exception('ERROR: illegal matrix size.')

    @staticmethod
    def __subtract(my_matrix: list, target: list):
        if len(my_matrix) == len(target) and len(my_matrix[0]) == len(target[0]):
            new_matrix: list = [[0.0] * len(target[0]) for i in range(len(target))]  # span a empty matrix
            temp_range: range = range(len(my_matrix[0]))
            for role in range(len(my_matrix)):
                for element in temp_range:
                    new_matrix[role][element] = my_matrix[role][element] - target[role][element]
            return new_matrix
        raise Exception('ERROR: illegal matrix size.')

    @staticmethod
    def __add(my_matrix: list, target: list):
        if len(my_matrix) == len(target) and len(my_matrix[0]) == len(target[0]):
            new_matrix: list = [[0.0] * len(target[0]) for i in range(len(target))]  # span a empty matrix
            temp_range: range = range(len(my_matrix[0]))
            for role in range(len(my_matrix)):
                for element in temp_range:
                    new_matrix[role][element] = my_matrix[role][element] + target[role][element]
            return new_matrix
        raise Exception('ERROR: illegal matrix size.')

    """
    elimination core below:
    """

    @staticmethod
    def __reduced_echelon_process_swap_zero_rows(my_matrix: list, target_row: int, column_number_count: int) -> list:
        """
detect for pivot zero. If there is a zero in the pivot, swap the rows and fix this. return my_matrix if
everything is fine and raise AllColumnZeroException if not possible swapping solution.
        @param my_matrix: input matrix (will be modified during execution)
        @param target_row: the row which the pivot exist
        @param column_number_count: the column that the pivot should exist (in a huge loop).
        @return: my_matrix
        """
        if my_matrix[target_row][column_number_count]:
            return my_matrix
        swaped: bool = False
        row_count: int
        for row_count in range(target_row + 1, len(my_matrix)):
            if my_matrix[row_count][column_number_count]:
                my_matrix[row_count], my_matrix[target_row] = my_matrix[target_row], my_matrix[row_count]
                swaped = True
        if not swaped:
            raise AllColumnZeroException
        return my_matrix

    @staticmethod
    def __re_turn_current_row_pivot_to_one(row: list, poz: int):
        """
turn the element in poz to one and subtract other elements
        @param row: input row (will return a new copied element)
        @param poz: the poz which one should exist
        @return: row list
        """
        poz_dev_num: float = row[poz]
        return [element / poz_dev_num for element in row]

    @staticmethod
    def __re_turn_the_whole_column_into_re_form(my_matrix: list, expected_pivot_row: int, column_number_count: int) -> (
            list, bool):
        """
turn the selected column in to reduced echelon (row reduced) form.
        @param my_matrix: the input matrix (will be modified)
        @param expected_pivot_row: the row which the pivot should occurred next, should +1 for the current pivot
        @param column_number_count: the column that the pivot should exist (in a huge loop).
        @return: (my_matrix: list, is_pivot: bool) my_matrix and the existence of the pivot in the current row.
        """
        # return the matrix step and the bool which indicate the changing statement of the matrix
        try:
            my_matrix: list = matrix.__reduced_echelon_process_swap_zero_rows(my_matrix, expected_pivot_row,
                                                                              column_number_count)
            # print(my_matrix)
            my_matrix[expected_pivot_row] = matrix.__re_turn_current_row_pivot_to_one(
                my_matrix[expected_pivot_row], column_number_count)
        except AllColumnZeroException:
            return my_matrix, False  # no pivot exist in this column and all the elements in the column are zero.
        except IndexError:
            raise Exception(f"An fatal error occurred in re_core: \n "
                            f"matrix :{my_matrix} {expected_pivot_row} {column_number_count}")

        for element_row in range(len(my_matrix)):
            poz_dev_num = my_matrix[element_row][column_number_count]
            if element_row != expected_pivot_row:
                # subtract all the rows
                for element_column in range(len(my_matrix[element_row])):
                    my_matrix[element_row][element_column] = my_matrix[element_row][element_column] - \
                                                             my_matrix[expected_pivot_row][element_column] * poz_dev_num
        return my_matrix, True

    # ---------------------------------------------

    @staticmethod
    def __null_space(my_matrix: list) -> list:
        """
return the null space of the my_matrix
        @param my_matrix: the input matrix
        @return: a list of list(vectors, not vector() object), which indicates the null space of the given matrix
        """
        # same as column space
        expected_pivot_row_count: int = 0
        pivot_index: list = []  # the index of pivots
        for column_number_count in range(min(len(my_matrix[0]), len(my_matrix))):
            my_matrix, has_pivot = matrix. \
                __re_turn_the_whole_column_into_re_form(my_matrix, expected_pivot_row_count, column_number_count)
            if has_pivot:
                pivot_index.append(column_number_count)
                expected_pivot_row_count += 1

        free_variable_index: set = set(range(len(my_matrix[0]))) - set(pivot_index)
        return_null_space: list = []
        for free_variable_column in free_variable_index:
            #  put the nagitive value of the column[free_variable_column] into a list
            current_solve_list = [- my_matrix[row_index][free_variable_column] for row_index in range(len(pivot_index))]
            final_result_list_index = 0
            current_null_space = []
            """
            put all the element in the individual null space vector by following rules:
            
            -> if the result_index: int is in pivot                           
                        append  current_solve_list[final_result_list_index]
            -> if the result_index: int is in the free_variable_column         
                        append 1 (for the x in the = 1, also is free_variable_column) or 0
            """
            for result_index in range(len(my_matrix[0])):  # Starting solving with a special one
                if result_index in pivot_index:
                    current_null_space.append(current_solve_list[final_result_list_index])
                    final_result_list_index += 1
                elif result_index == free_variable_column:
                    current_null_space.append(1.0)
                else:
                    current_null_space.append(0)

            return_null_space.append(current_null_space)
        return return_null_space

    def null_space(self):
        return [vector(content) for content in self.__null_space(copy.deepcopy(self.matrix))]

    def left_null_space(self):
        #  I DO NOT THINK TO WRITE LEFT NONE SPACE IS A GOOD IDEA. I'D BETTER USE NULL SPACE TRANSPOSE.
        return [vector(content) for content in self.__null_space(copy.deepcopy(self.transpose().matrix))]

    @staticmethod
    def __gauss_jordan_elimination(my_matrix: list):
        """
regards the input matrix as an function set and offer a solution by useing gauss jordan elimination.
        @param my_matrix: the matrix which you want to solve
        @return: my_matrix: list[list]
        """
        row, column = len(my_matrix), len(my_matrix[0])
        step_number = min(row, column - 1)
        for column_row_count in range(step_number):
            my_matrix, has_pivot = matrix.__re_turn_the_whole_column_into_re_form(my_matrix, column_row_count,
                                                                                  column_row_count)
            if not has_pivot:
                return my_matrix
        return my_matrix

    def gauss_jordan_elimination(self):
        return matrix(self.__gauss_jordan_elimination(copy.deepcopy(self.matrix)))

    @staticmethod
    def __reduced_echelon_form(my_matrix: list) -> list:
        """
provides the row reduced form of the given matrix.
        @param my_matrix: the matrix which you want to solve
        @return: my_matrix: list[list]
        """
        expected_pivot_row_count = 0
        for column_number_count in range(min(len(my_matrix[0]), len(my_matrix))):
            my_matrix, has_pivot = matrix. \
                __re_turn_the_whole_column_into_re_form(my_matrix, expected_pivot_row_count, column_number_count)
            if has_pivot:
                expected_pivot_row_count += 1
        return my_matrix

    def reduced_echelon_form(self):
        my_matrix = self.__reduced_echelon_form(copy.deepcopy(self.matrix))
        return matrix(my_matrix)

    @staticmethod
    def __column_space(my_matrix: list) -> list:
        """
return the index of the column space of the given matrix
        @param my_matrix: the matrix which you want to solve
        @return: the index of the column space in my_matrix
        """
        # return the list of the index of the matrix
        expected_pivot_row_count: int = 0
        pivot_index: list = []
        for column_number_count in range(min(len(my_matrix[0]), len(my_matrix))):
            my_matrix, has_pivot = matrix. \
                __re_turn_the_whole_column_into_re_form(my_matrix, expected_pivot_row_count, column_number_count)
            if has_pivot:
                pivot_index.append(column_number_count)
                expected_pivot_row_count += 1
        return pivot_index

    def column_space(self, return_index: bool = False):
        pivot_index = self.__column_space(copy.deepcopy(self.matrix))
        if return_index:
            return set(pivot_index)
        else:
            return [vector(self.__get_column(self.matrix, index)) for index in pivot_index]

    def row_space(self, return_index: bool = False):
        return self.transpose().column_space(return_index)

    """
    independent method below
    """

    @staticmethod
    def __transpose(my_matrix: list):
        return [[my_matrix[row_number][column_number] for row_number in range(0, len(my_matrix))] for column_number
                in range(0, len(my_matrix[0]))]

    def transpose(self):
        return matrix(self.__transpose(copy.deepcopy(self.matrix)))

    @staticmethod
    def __invert(my_matrix: list):
        row, column = len(my_matrix), len(my_matrix[0])
        if row != column:
            print('the matrix must be square matrix.')
            return None
        my_matrix = matrix.__combine(my_matrix, matrix.__identity_matrix(row))
        for count in range(row):
            return_matrix, has_pivot = matrix.__re_turn_the_whole_column_into_re_form(
                my_matrix, count, count)
            if not has_pivot:
                print('the matrix is singular. It is un invertible.')
                return None
        for line in range(row):
            my_matrix[line] = my_matrix[line][row:]
        return my_matrix

    def invert(self):
        return matrix(self.__invert(copy.deepcopy(self.matrix)))

    @staticmethod
    def __combine(my_matrix: list, target: list) -> list:
        my_matrix = [my_matrix[count] + target[count] for count in range(len(my_matrix))]  # MAKE SURE THAT MY_MATRIX
        # OBJECT HAS MODIFIED
        return my_matrix

    def combine(self, target):
        return matrix(self.__combine(copy.deepcopy(self.matrix), target.matrix))

    @staticmethod
    def __cast_to_complex(my_matrix: list, target: list) -> list:
        """
input two matrices(only contain real elements)
        @param my_matrix:
        @param target:
        @return:
        """
        if (len(my_matrix), len(my_matrix[0])) != (len(target), len(target[0])):
            raise LengthError("The row and column must be the same in both matrix.")
        for row_count in range(len(my_matrix)):
            for column_count in range(len(my_matrix[0])):
                my_matrix[row_count][column_count] = \
                    complex(my_matrix[row_count][column_count], target[row_count][column_count])
        return my_matrix

    def cast_to_complex(self, target):
        return matrix(self.__cast_to_complex(copy.deepcopy(self.matrix), target.matrix))

    @staticmethod
    def __conjugate(my_matrix: list) -> list:
        for row_index in range(len(my_matrix)):
            for column_index in range(len(my_matrix[0])):
                if type(my_matrix[row_index][column_index]) is complex:
                    my_matrix[row_index][column_index] = my_matrix[row_index][column_index].conjugate()
        return my_matrix

    def conjugate(self):
        return matrix(copy.deepcopy(self.matrix))

    def conjugate_transpose(self):
        return matrix(self.__transpose(copy.deepcopy(self.matrix)))

    @staticmethod
    def __round_to_square_matrix(my_matrix: list):
        row_number, column_number = len(my_matrix), len(my_matrix[0])
        if row_number == column_number:
            return my_matrix
        elif row_number > column_number:
            diff = row_number - column_number
            return matrix.__combine(my_matrix, [[0] * diff for x in range(len(my_matrix))])
        else:
            diff, temp_len = column_number - row_number, len(my_matrix[0])
            for x in range(diff):
                my_matrix.append([0] * temp_len)
            return my_matrix

    def round_to_square_matrix(self):
        return matrix(self.__round_to_square_matrix(copy.deepcopy(self.matrix)))

    @staticmethod
    def __get_projection_matrix(my_matrix: list):
        pivot_index: list = matrix.__column_space(copy.deepcopy(my_matrix))
        my_matrix: list = [[my_matrix[row_count][column_count] for column_count in pivot_index] for row_count in
                           range(len(
                               my_matrix))]
        # A * (A.transpose() * A).invert() * A.transpose()
        A_transpose: list = matrix.__transpose(copy.deepcopy(my_matrix))
        A_transpose_A = matrix.__multiply(copy.deepcopy(A_transpose), my_matrix)
        A_transpose_A_inv = matrix.__invert(A_transpose_A)
        return matrix.__multiply(matrix.__multiply(my_matrix, A_transpose_A_inv), A_transpose)

    def get_projection_matrix(self):
        return matrix(self.__get_projection_matrix(copy.deepcopy(self.matrix)))

    @staticmethod
    def __project(my_matrix: list, target: list, to_orthogonal_space: bool = False):
        P = matrix.__get_projection_matrix(copy.deepcopy(target))
        if to_orthogonal_space:
            P = matrix.__subtract(matrix.__identity_matrix(len(P)), P)
        return matrix.__multiply(P, my_matrix)

    def project(self, target, to_orthogonal_space: bool = False):
        return matrix(self.__project(self.matrix, copy.deepcopy(target.matrix), to_orthogonal_space))

    @staticmethod
    def least_square(data: list):
        # A^T * A * X = A^T * b
        A = []
        b = []
        for a in data:
            A.append([a[0], 1])
            b.append(a[1])
        A, b = matrix(A), vector(b)
        ltemp = A.transpose() * A
        rtemp = A.transpose() * b
        solution_matrix = ltemp.combine(rtemp)
        solved_matrix = solution_matrix.gauss_jordan_elimination()
        return f"\nleast square of {data} for y = Cx + D: \n C = {solved_matrix.matrix[0][-1]}," \
               f" D = {solved_matrix.matrix[1][-1]} \n"

    """
    matrix creation
    """

    @staticmethod
    def __identity_matrix(size: int = 4) -> list:
        return_matrix = [[0] * size for i in range(size)]
        for count in range(size):
            return_matrix[count][count] = 1
        return return_matrix

    @staticmethod
    def identity_matrix(size: int = 4):
        return matrix(matrix.__identity_matrix(size))

    @staticmethod
    def __zero_matrix(row: int = 4, col: int = 4) -> list:
        return [[0] * col for i in range(row)]

    @staticmethod
    def zero_matrix(row: int = 4, col: int = 4):
        return matrix([[0] * col for i in range(row)])

    @staticmethod
    def __random_matrix_int(row: int = 4, col: int = 4, max_: int = 10, min_: int = 0) -> list:
        return_matrix = [[0] * col for i in range(row)]
        for row_count in range(row):
            for column_count in range(col):
                return_matrix[row_count][column_count] = random.randint(min_, max_)
        return return_matrix

    @staticmethod
    def __random_matrix_float(row: int = 4, col: int = 4, max_: float = 10, min_: float = 0) -> list:
        return_matrix = [[0] * col for i in range(row)]
        for row_count in range(row):
            for column_count in range(col):
                return_matrix[row_count][column_count] = random.uniform(min_, max_)
        return return_matrix

    @staticmethod
    def random_matrix(row: int = 4, col: int = 4, max_: Union[float, int, complex] = 10,
                      min_: Union[float, int, complex] = 0, type_: Union[type, tuple] = int, seed: Any = None):
        if seed is not None:
            random.seed(seed)
        if type_ == int:
            return matrix(matrix.__random_matrix_int(row, col, max_, min_))
        elif type_ == float:
            return matrix(matrix.__random_matrix_float(row, col, max_, min_))
        elif type_ == (complex, int):
            if type(max_) is complex:
                real_max, imag_max, real_min, imag_min = max_.real, max_.imag, min_.real, min_.imag
            else:
                real_max, imag_max, real_min, imag_min = max_, max_, min_, min_
            my_matrix1, my_matrix2 = \
                (matrix.__random_matrix_int(row, col, real_max, real_min),
                 matrix.__random_matrix_int(row, col, imag_max, imag_min))
            return matrix(matrix.__cast_to_complex(my_matrix1, my_matrix2))
        elif type_ == (complex, float):
            if type(max_) is complex:
                real_max, imag_max, real_min, imag_min = max_.real, max_.imag, min_.real, min_.imag
            else:
                real_max, imag_max, real_min, imag_min = max_, max_, min_, min_
            my_matrix1, my_matrix2 = \
                (matrix.__random_matrix_float(row, col, real_max, real_min),
                 matrix.__random_matrix_float(row, col, imag_max, imag_min))
            return matrix(matrix.__cast_to_complex(my_matrix1, my_matrix2))
        else:
            raise TypeError("only support int and float for random.")

    def to_vector(self, split_index: Union[range, list, tuple, set, frozenset, None] = None) -> list:
        """
        to split the given matrix to multiple vectors

        @param split_index: determines the index of the matrix. If split_index is None, then the method will split all the
        element in the matrix.

        @return a list of vector object.
        """
        if split_index is None:
            split_index = range(0, len(self.matrix[0]))
        return [vector([row[index] for row in self.matrix]) for index in split_index]

    def to_list(self):
        return copy.deepcopy(self.matrix)

    def to_str(self):
        return str(self.matrix)


class vector(matrix):
    def __init__(self, input_matrix: Union[list, None]):
        self.matrix = input_matrix
        if self.matrix is None:
            return
        if type(self.matrix[0]) != list:
            self.matrix = [self.matrix]
        if len(self.matrix) != 1 and len(self.matrix[0]) != 1:
            raise TypeError('Use matrix instead.')
        self.matrix = [[tmp_loop] for tmp_loop in self.matrix[0]]

    def __getitem__(self, item):  # make it iterable.
        return self.matrix[item]

    def __repr__(self):
        if self.matrix:
            return "vector: \n" + str([round_all(element[0]) for element in self.matrix]) + '\n'
        return "vector: \n NONE VECTOR\n"

    @staticmethod
    def __dot(my_matrix: list, target: list) -> int or float:
        return sum([my_matrix[temp_loop][0] * target[temp_loop][0] for temp_loop in range(len(my_matrix))])

    def dot(self, target: matrix) -> int or float:
        if len(self.matrix) != len(target.matrix):
            raise LengthError("the length of the vector is different!")
        return vector.__dot(self.matrix, target.matrix)

    @staticmethod
    def __module(my_matrix: list) -> int or float:
        return sum([element[0] ** 2 for element in my_matrix]) ** 0.5

    def module(self) -> int or float:
        return self.__module(self.matrix)

    """other methods below"""

    @staticmethod
    def random_vector(length: int = 3, max_: Union[float, int, complex] = 10,
                      min_: Union[float, int, complex] = 0, type_: Union[type, tuple] = int, seed: Any = None):
        return vector(matrix.random_matrix(length, 1, max_, min_, type_, seed).to_vector())

    def to_matrix(self):
        return matrix(self.matrix)

    def to_list(self):
        return [element[0] for element in self.matrix]

    def to_str(self):
        return str([element[0] for element in self.matrix])


"""
other
"""


def copyright():
    print(COPYRIGHT_CONTENT)


def about():
    print(ABOUT_CONTENT)


def command_prompt():
    print('this is a basic compute software of linear algebra developed by Wang Weizheng')
    print("For more information, type 'copyright()' or 'about()' in the console.\n")
    while True:
        try:
            input_tempstr = input(">>> ")
            if input_tempstr in locals().keys():
                print(locals()[input_tempstr])
                continue
            else:
                exec(input_tempstr)
        except SystemExit:
            raise SystemExit
        except Exception as e:
            print(f"an error occurred during the execution, {e}")


if __name__ == '__main__':
    command_prompt()  # you could run your own code by commenting this sentence
    """type your own code below. For example:"""
