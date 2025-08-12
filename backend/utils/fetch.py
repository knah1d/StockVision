import random
import time

# --- Game Data ---
rooms = [
    "a dusty library filled with cobwebs",
    "a torch-lit hallway with eerie whispers",
    "a treasure room glittering with gold",
    "a cold damp cell with rusted bars",
    "a grand hall with a cracked throne",
    "a kitchen with rotten food and broken utensils",
]

monsters = ["Goblin", "Skeleton", "Orc", "Giant Spider", "Ghost"]
treasures = ["gold coins", "a diamond sword", "a healing potion", "a magical amulet"]

inventory = []
health = 100


def slow_print(text, delay=0.03):
    for ch in text:
        print(ch, end="", flush=True)
        time.sleep(delay)
    print()


def combat(monster):
    global health
    slow_print(f"A {monster} appears! 🐉")
    while True:
        choice = input("Do you want to (f)ight or (r)un? ").lower()
        if choice == "f":
            player_attack = random.randint(10, 30)
            monster_attack = random.randint(5, 20)
            slow_print(f"You strike the {monster} for {player_attack} damage!")
            slow_print(f"The {monster} hits you for {monster_attack} damage!")
            health -= monster_attack
            if health <= 0:
                slow_print("You have been defeated! 💀")
                return False
            if random.choice([True, False]):
                slow_print(f"You defeated the {monster}! 🏆")
                return True
        elif choice == "r":
            if random.choice([True, False]):
                slow_print("You managed to escape! 🏃")
                return True
            else:
                slow_print("You failed to escape! The monster attacks!")
                health -= random.randint(10, 25)
                if health <= 0:
                    slow_print("You have been slain! 💀")
                    return False
        else:
            slow_print("Invalid choice!")


def explore_room():
    global health
    room = random.choice(rooms)
    slow_print(f"You enter {room}.")

    # Random events
    event = random.choice(["monster", "treasure", "empty", "trap"])
    if event == "monster":
        if not combat(random.choice(monsters)):
            return False
    elif event == "treasure":
        item = random.choice(treasures)
        inventory.append(item)
        slow_print(f"You found {item}! 🪙")
        if item == "a healing potion":
            health = min(100, health + 30)
            slow_print(f"You drink the potion. Health restored to {health} ❤️.")
    elif event == "trap":
        damage = random.randint(10, 30)
        health -= damage
        slow_print(f"A hidden trap injures you! You lose {damage} HP.")
        if health <= 0:
            slow_print("You have died from your wounds. 💀")
            return False
    else:
        slow_print("The room is empty... eerie silence surrounds you.")

    return True


def main():
    slow_print("🏰 Welcome to ESCAPE THE DUNGEON 🏰")
    slow_print("Find your way out alive, or perish within...")
    slow_print(f"Your starting health: {health} ❤️\n")

    turns = 0
    while health > 0:
        choice = input("Do you want to (e)xplore or (q)uit? ").lower()
        if choice == "e":
            turns += 1
            if not explore_room():
                break
            # Chance to find the exit
            if random.randint(1, 6) == 1:
                slow_print("You found the exit door! 🚪")
                if inventory:
                    slow_print(f"You escape with your loot: {', '.join(inventory)}")
                else:
                    slow_print("You escape, but with empty hands.")
                slow_print(f"Survived {turns} rooms explored. ")
                break
        elif choice == "q":
            slow_print("You give up and accept your fate in the dungeon...")
            break
        else:
            slow_print("Invalid choice.")

    slow_print("\nGame Over. Thanks for playing! 🎮")


if __name__ == "__main__":
    main()
