# File: ga_llm/engine.py
# Author: Jonas Lin, Jacky Cen
# Version: 0.2
# Description: Hybrid GA-LLM engine with detailed execution logging

import random
from typing import List, Tuple, Callable, Type
from concurrent.futures import ThreadPoolExecutor, as_completed
from ga_llm.base_gene import BaseGene
from ga_llm.config import EvolutionConfig
from ga_llm.llm_utils import default_llm_call
from ga_llm.logging_utils import setup_logger

class HybridEvolutionEngine:
    """Main engine for LLM-guided evolutionary optimization."""

    def __init__(
        self,
        config: EvolutionConfig,
        gene_cls: Type[BaseGene],
        task_prompt: str,
        eval_prompt: str,
        llm_callback: Callable[[str], str] = None,
        logger=None
    ):
        self.config = config
        self.gene_cls = gene_cls
        self.task_prompt = task_prompt
        self.eval_prompt = eval_prompt
        self.llm = llm_callback or default_llm_call
        self.logger = logger or setup_logger("evolution")

    def initialize_population(self) -> List[BaseGene]:
        """Generate initial population using LLM."""
        self.logger.info("Initializing population, target size: %d", self.config.population_size)
        population = []
        prompt = f"{self.task_prompt}\n\nReturn ONLY raw JSON. No markdown, no explanation."

        def create_gene(idx: int):
            try:
                response = self.llm(prompt)
                gene = self.gene_cls(llm_engine=self.llm)
                gene.parse_from_text(response)
                self.logger.debug(f"Individual #{idx} generated successfully:\n{gene.to_text()}")
                return gene
            except Exception as e:
                self.logger.warning(f"Individual #{idx} generation failed: {e}")
                return None

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(create_gene, i + 1) for i in range(self.config.population_size)]
            for future in as_completed(futures):
                gene = future.result()
                if gene:
                    population.append(gene)

        return population

    def evaluate(self, gene: BaseGene) -> float:
        """Evaluate an individual using LLM score prompt."""
        text = gene.to_text()
        prompt = f"{self.eval_prompt}\n\nProposal to evaluate:\n{text}\n\nReturn ONLY score [0-10]"
        try:
            score = float(self.llm(prompt).strip().replace("[", "").replace("]", ""))
            self.logger.debug("Individual score: %.2f", score)
            return min(max(score, 0.0), 10.0)
        except Exception as e:
            self.logger.warning("Evaluation failed, returning 0.0: %s", e)
            return 0.0

    def evolve(self) -> Tuple[BaseGene, float]:
        """Main evolution loop."""
        population = self.initialize_population()
        best_solution = None
        best_score = -float("inf")
        no_improvement_rounds = 0

        for generation in range(self.config.max_generations):
            self.logger.info("==> Starting generation %d", generation + 1)
            self.logger.info("Starting parallel population evaluation, total %d individuals", len(population))

            def _evaluate_indiv(indiv):
                try:
                    score = self.evaluate(indiv)
                    self.logger.debug("Individual score: %.2f", score)
                    return (indiv, score)
                except Exception as e:
                    self.logger.warning("Scoring failed, defaulting to 0: %s", e)
                    return (indiv, 0.0)

            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(_evaluate_indiv, indiv) for indiv in population]
                scored = [future.result() for future in as_completed(futures)]

            scored.sort(key=lambda x: x[1], reverse=True)

            top_gene, top_score = scored[0]
            self.logger.info("Best score of generation %d: %.2f", generation + 1, top_score)

            if top_score > best_score:
                best_solution, best_score = top_gene, top_score
                no_improvement_rounds = 0
                self.logger.info("🔼 Best solution updated, current top score: %.2f", best_score)
            else:
                no_improvement_rounds += 1
                self.logger.info("No improvement rounds: %d/%d", no_improvement_rounds, self.config.early_stopping_rounds)
                if no_improvement_rounds >= self.config.early_stopping_rounds:
                    self.logger.warning("🎯 Early stopping condition triggered, evolution terminated.")
                    break

            elite_count = max(1, int(len(population) * self.config.elite_ratio))
            elites = [indiv for indiv, _ in scored[:elite_count]]
            self.logger.info("Number of elites retained: %d", elite_count)

            next_gen = elites[:]

            while len(next_gen) < self.config.population_size:
                p1 = self._select(scored)
                p2 = self._select(scored)
                self.logger.debug("Selecting parents for crossover")

                if self.config.use_llm_for_crossover:
                    child = self._llm_crossover(p1, p2)
                else:
                    child = p1.crossover(p2)

                if random.random() < self.config.mutation_rate:
                    self.logger.debug("Individual will undergo mutation")
                    child = child.mutate()

                next_gen.append(child)

            population = next_gen
            self.logger.info("Generation %d completed", generation + 1)

        self.logger.info("✅ Evolution completed, best solution score: %.2f", best_score)
        return best_solution, best_score

    def _select(self, scored: List[Tuple[BaseGene, float]], k: int = 3) -> BaseGene:
        """Tournament selection strategy."""
        candidates = random.sample(scored, k)
        return max(candidates, key=lambda x: x[1])[0]

    def _llm_crossover(self, p1: BaseGene, p2: BaseGene) -> BaseGene:
        """Combine two genes and improve the offspring with LLM."""
        self.logger.debug("Performing LLM-enhanced crossover")
        temp_gene = p1.crossover(p2)
        merged_text = f"""Synthesize the best from the two candidates below.

        Candidate A:
        {p1.to_text()}

        Candidate B:
        {p2.to_text()}

        {self.task_prompt}

        Output ONLY the improved JSON.
        No markdown, no explanation.
        """
        try:
            improved = self.llm(merged_text)
            new_gene = self.gene_cls(llm_engine=self.llm)
            new_gene.parse_from_text(improved)
            self.logger.debug("Crossover offspring generated successfully:\n%s", new_gene.to_text())
            return new_gene
        except Exception as e:
            self.logger.warning("LLM crossover failed, using fallback offspring: %s", e)
            return temp_gene
