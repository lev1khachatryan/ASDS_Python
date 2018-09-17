import math
def check_prime():
    num = int(input("Enter a number: "))
    if num > 1:
        for i in range(2, int(math.sqrt(num))):
            if (num % i) == 0:
                print(num, "is not a prime number")
                break
        else:
            print(num, "is a prime number")
    else:
        print(num, "is not a prime number")
    again()

def again():
    check_again = input('''
Do you want to check prime number again?
Please type Y for YES or N for NO.
''')

    if check_again.upper() == 'Y':
        check_prime()
    elif check_again.upper() == 'N':
        print('See you later.')
    else:
        again()
check_prime()