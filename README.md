# RL_Signaling

World is composed of states coming from n_features=3 binary random variables.

The procedure is the following:

Step 0:
- World emits some random state. Say with features X,Y,Z are binary random variables.
- Agents observe different substates of that state. For exampe, agents of type 1 observe X,Y and agents of type 2 observe Y,Z.
- Agents emit some signals to each other, say from a set S={s1,s2}. These are actions with their associated q_table. No reward in this step.

Step 1:
- Agents receive signal from others, and together with their respective substates they have a new input for action.
- Agents perform actions from set A. Assume that they are playing the same game and that their payoffs do not depend on each others actions, but do depend on the whole state (not just the substate+signal that they as input). In our example, G:XxYxZxA→ℝ. Future setups might be such that payoff depends also on actions, so G:XxYxZxA1xA2→ℝ.
- Agents get rewards from the environment.
- Agents update both q_tables, the one for signals and the one for actions. In the future these might be neural nets.


Observation 1: Agents payoffs are not immediately correlated, they are independent of the others actions and payoffs.


Observation 2: This means that in principle they have no incentive to communicate meaningfully.


Hypothesis: Despite Observation 1, there will be region of the space in which they coordinate. This means something like the fact that the signal output and signal decoding will match the true (hidden for receiver) state of the world.

# Tareas

- Hacer el subsetting the quien observa que variables al azar.
- Reporta para cada step en el proceso, la MI y NMI.
- TENES TRES NIVELES:
  1. Full information (both agents observe the world, and there is no signaling)
  2. Partial information with out signals.
  3. Partial information with signals.
  4. Al final tenes que plotear, para los mismos meta parameters (que variables observa quien, que game), la distribucion de gain in reward and gain in information that adding singaling involved. Algo asi como un historgram, porque seguro que es medio normal.

Cuidado!
- Ojo que cada vez que inicializas el environment te da un game at random. Capaz los games tienen que ser determinados desde afuera. Porque queres las tres simulaciones con el mismo juego para los jugadores.

Hecho:
- Distintos juegos para cada agente.
- Agents observe variables at random
