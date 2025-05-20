def base_reward(pnl_diff, inventory, final=False, max_inventory=100, zeta=0, eta=0.01):
    """
    Core reward function:

    Non-terminal step:
        R_n = ΔPnL_n - zeta * (inventory / max_inventory)^2

    Terminal step:
        R_T = ΔPnL_T - (zeta + eta) * (inventory / max_inventory)^2

    This encourages agents to:
    - Stay close to zero inventory during training
    - Liquidate positions before episode end (via eta penalty)

    Parameters:
    - pnl_diff: change in mark-to-market portfolio value
    - inventory: current inventory level
    - final: whether this is the terminal timestep
    - max_inventory: normalization bound for inventory
    - zeta: running inventory penalty coefficient
    - eta: terminal inventory penalty coefficient

    Returns:
    - reward: float
    """
    inv_ratio = inventory / max_inventory
    penalty = zeta * (inv_ratio ** 2)
    if final:
        penalty += eta * (inv_ratio ** 2)
    return pnl_diff - penalty

def Adversary_reward(*args, **kwargs):
    """
    Adversary or attacker: tries to reduce agent performance (zero-sum)
    """
    return -base_reward(*args, **kwargs)

def MMA_reward(*args, **kwargs):
    return base_reward(*args, **kwargs)

def MMB1_reward(*args, **kwargs):
    return base_reward(*args, **kwargs)

def MMB2_reward(*args, **kwargs):
    '''
    To reflect the strategic opacity of real-world market participants, agent B2 is only rewarded based on agent A’s profit dynamics, but does not observe any internal variables of A (e.g. inventory or cash).
    This setup forms a realistic zero-sum market game under asymmetric information.
    '''
    return -base_reward(*args, **kwargs)
