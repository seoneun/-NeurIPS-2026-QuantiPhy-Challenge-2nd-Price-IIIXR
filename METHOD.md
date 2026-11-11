# Method (Track B, team IIIXR Lab)

Our final method (which produces the CSV file) is a composition of a code world model and (frozen) open-weight models.
The development process of the method, which might be more important, is based on decentralized multi-agent code world model generation. Several coding agents each develop their own method: a
program that models the physical scene of a question and computes the asked quantity from it. No central agent coordinates
their work.

1. Independent development. Each agent develops its method on its own. During development the agents do not communicate with
   each other, so each agent builds its method from its own attempts.
2. Propagation. When an agent expects that a new method reaches a higher performance than its previous methods, it propagates
   the new method to the other agents: the code and a description of the method.
3. Adoption. An agent that receives a method can take it into its own work and continue its development from it. A method that
   improves performance spreads across the agents in this way; a method that does not stays with the agent that wrote it.
4. Output. The final method is a set of agent-generated programs.


Declaration of closed-API usage during development. During development, each agent was powered by Claude Opus 5.5. We used a private GitHub repository as the communication method for multi-agent communication.
