from random import randint, random


def new_game(n):
    matrix = []

    for i in range(n):
        matrix.append([0] * n)
    return matrix


def add_two_or_four(mat):
    a = randint(0, len(mat)-1)
    b = randint(0, len(mat)-1)
    while(mat[a][b] != 0):
        a = randint(0, len(mat)-1)
        b = randint(0, len(mat)-1)
    mat[a][b] = [2, 4][(0 if random() < .9 else 1)]
    return mat


def game_state(mat):
    # for i in range(len(mat)):
    #     for j in range(len(mat[0])):
    #         if mat[i][j] == 8192:
    #             return 'win'
    for i in range(len(mat)-1):
        for j in range(len(mat[0])-1):
            if mat[i][j] == mat[i+1][j] or mat[i][j+1] == mat[i][j]:
                return 'not over'
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            if mat[i][j] == 0:
                return 'not over'
    for k in range(len(mat)-1):
        if mat[len(mat)-1][k] == mat[len(mat)-1][k+1]:
            return 'not over'
    for j in range(len(mat)-1):
        if mat[j][len(mat)-1] == mat[j+1][len(mat)-1]:
            return 'not over'
    return 'lose'


def reverse(mat):
    new = []
    for i in range(len(mat)):
        new.append([])
        for j in range(len(mat[0])):
            new[i].append(mat[i][len(mat[0])-j-1])
    return new


def transpose(mat):
    new = []
    for i in range(len(mat[0])):
        new.append([])
        for j in range(len(mat)):
            new[i].append(mat[j][i])
    return new


def cover_up(mat):
    size = len(mat)
    new = [[0]*size for _ in range(size)]
    done = False
    for i in range(size):
        count = 0
        for j in range(size):
            if mat[i][j] != 0:
                new[i][count] = mat[i][j]
                if j != count:
                    done = True
                count += 1
    return (new, done)

# mat = [[1,4,0,0],[5,8,7,8],[7,5,11,0],[13,10,15,0]]
# new, done = cover_up(mat)
# print(done)
# [print(row) for row in mat]
# print("")
# [print(row) for row in new]


def merge(mat):
    size = len(mat)
    done = False
    score_increase = 0
    for i in range(size):
        for j in range(size - 1):
            if mat[i][j] == mat[i][j+1] and mat[i][j] != 0:
                mat[i][j] *= 2
                mat[i][j+1] = 0
                done = True
                score_increase += mat[i][j]

    return (mat, done, score_increase)


def up(game):
        # print("up")
        # return matrix after shifting up
        game = transpose(game)
        game, done = cover_up(game)
        temp = merge(game)
        game = temp[0]
        done = done or temp[1]
        game = cover_up(game)[0]
        game = transpose(game)
        return (game, done, temp[2])


def down(game):
        # print("down")
        game = reverse(transpose(game))
        game, done = cover_up(game)
        temp = merge(game)
        game = temp[0]
        done = done or temp[1]
        game = cover_up(game)[0]
        game = transpose(reverse(game))
        return (game, done, temp[2])


def left(game):
        # print("left")
        # return matrix after shifting left
        game, done = cover_up(game)
        temp = merge(game)
        game = temp[0]
        done = done or temp[1]
        game = cover_up(game)[0]
        return (game, done, temp[2])


def right(game):
        # print("right")
        # return matrix after shifting right
        game = reverse(game)
        game, done = cover_up(game)
        temp = merge(game)
        game = temp[0]
        done = done or temp[1]
        game = cover_up(game)[0]
        game = reverse(game)
        return (game, done, temp[2])
