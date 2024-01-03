from game_state import GameState


def main():
    state = GameState()
    state.slp(2)

    state.open_free_play()
    state.set_settings(19, 9, False, False)
    state.start_game()
     
if __name__ == '__main__':
    main()
