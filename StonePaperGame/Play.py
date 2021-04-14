import random

print()
print('Welcome to Stone-Papar-Scissors Game!! :)')
print()
num = int(input('How many Rounds you want to play? : '))
print()

i = 0
computer_score = 0
player_score = 0
options = ['S','c','P','s','p','C']
dict = {'S':'Stone', 'P':'Paper', 'C':'Scissors'}
draw = 0

print('Expected Inputs:')
print("     'S' for Stone")
print("     'P' for Paper")
print("     'C' for Scissors")
print()

while i < num:
    i = i+1

    inpt = input('Choose your option: ').capitalize()
    if inpt in options:
        print()
        pass
    else:
        print('Give an Expected Input.')
        print('Try Again!')
        print()
        continue

    comp_player = random.choice(options).capitalize()
    print("Your choice: " + dict[inpt])
    print("Computer's choice: " + dict[comp_player])

    if inpt == comp_player:
        draw = draw +1
        print("It's a Draw!")

    elif inpt == 'S':
        if comp_player == 'P':
            computer_score = computer_score +1
            print('Oh! You Loose. :(')

        elif comp_player == 'C':
            player_score = player_score +1
            print('Yeh! You Win. ;)')

    elif inpt == 'P':
        if comp_player == 'C':
            computer_score = computer_score +1
            print('Oh! You Loose. :(')

        elif comp_player == 'S':
            player_score = player_score +1
            print('Yeh! You Win. ;)')

    elif inpt == 'C':
        if comp_player == 'S':
            computer_score = computer_score +1
            print('Oh! You Loose. :(')
        elif comp_player == 'P':
            player_score = player_score +1
            print('Yeh! You Win. ;)')

    else:
        pass
    print()

print('Results:')
if computer_score < player_score:
    print('Congo! You win the match. :D')
elif computer_score > player_score:
    print('Sad! You Loose the match.')
else:
    print('Draw!')

print()
print('Final Scores:')
print('     Your Score: ' + str(player_score))
print("     Computer's Score: " + str(computer_score))
print('     Draw rounds: ' + str(draw))
print()
print('Thanks for playing..')
