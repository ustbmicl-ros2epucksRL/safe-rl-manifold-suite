PYTHON ?= python3
GS ?= gs

FIGURE_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PAPER_ROOT := $(abspath $(FIGURE_ROOT)/..)
REPO_ROOT := $(abspath $(PAPER_ROOT)/../..)
SAFE_RL := $(REPO_ROOT)/safe-rl-2027
PAPER_DATA := $(PAPER_ROOT)/data
MAIN_DIR := $(FIGURE_ROOT)/main
SUPPLEMENT_DIR := $(FIGURE_ROOT)/supplement
SCRIPT_DIR := $(FIGURE_ROOT)/scripts

MAIN_FIGURE := $(MAIN_DIR)/aber_lr_full_matrix.pdf
FAILURE_FIGURE := $(SUPPLEMENT_DIR)/aber_lr_failure_landscape.pdf
RECOVERY_FIGURE := $(SUPPLEMENT_DIR)/aber_lr_recovery_mechanism.pdf
INTERVENTION_FIGURE := $(SUPPLEMENT_DIR)/aber_intervention_event_audit.pdf
CONTRACT_FIGURE := $(SUPPLEMENT_DIR)/aber_safetygym_contract_v1.pdf
FASTPATH_FIGURE := $(SUPPLEMENT_DIR)/aber_fastpath_performance.pdf
ACTIVE_PDFS := $(MAIN_FIGURE) $(FAILURE_FIGURE) $(RECOVERY_FIGURE) $(INTERVENTION_FIGURE) $(CONTRACT_FIGURE) $(FASTPATH_FIGURE)

.PHONY: active main falsification failure recovery intervention contract fastpath verify-active verify-falsification

# These are exactly the six PDFs included by the current main paper and supplement.
active: main failure recovery intervention contract fastpath

main:
	$(PYTHON) "$(SCRIPT_DIR)/generate_aber_lr_figures.py" \
		--figure full-matrix --output-dir "$(MAIN_DIR)"

falsification:
	$(PYTHON) "$(SCRIPT_DIR)/generate_aber_lr_falsification_chain.py" \
		--output-dir "$(MAIN_DIR)"

failure:
	$(PYTHON) "$(SCRIPT_DIR)/generate_aber_lr_failure_landscape.py" \
		--output-dir "$(SUPPLEMENT_DIR)"

recovery:
	$(PYTHON) "$(SCRIPT_DIR)/generate_aber_lr_figures.py" \
		--figure recovery-mechanism --output-dir "$(SUPPLEMENT_DIR)"

intervention:
	$(PYTHON) "$(SAFE_RL)/experiments/aaai/analyze_aber_interventions.py" \
		--root "$(SAFE_RL)/results/aaai27_aber_paired_matrix" \
		--episodes-per-checkpoint 250 --step-seconds 0.02 \
		--json-output "$(PAPER_DATA)/aber_intervention_event_audit.json" \
		--csv-output "$(PAPER_DATA)/aber_intervention_event_audit.csv" \
		--figure-prefix "$(SUPPLEMENT_DIR)/aber_intervention_event_audit"

contract:
	$(PYTHON) "$(SAFE_RL)/experiments/aaai/plot_aber_safetygym_contract_v1.py" \
		--audit "$(PAPER_DATA)/aber_safetygym_contract_v1.json" \
		--output-prefix "$(SUPPLEMENT_DIR)/aber_safetygym_contract_v1"

fastpath:
	$(PYTHON) "$(SAFE_RL)/experiments/aaai/plot_aber_fastpath.py" \
		--heldout "$(PAPER_DATA)/aber_fastpath_heldout.json" \
		--runtime "$(PAPER_DATA)/aber_fastpath_runtime.json" \
		--output-prefix "$(SUPPLEMENT_DIR)/aber_fastpath_performance"

verify-active: active
	@for figure in $(ACTIVE_PDFS); do \
		test -s "$$figure"; \
		outlined="$$figure.outlined.pdf"; \
		$(GS) -q -dBATCH -dNOPAUSE -dSAFER -sDEVICE=pdfwrite \
			-dCompatibilityLevel=1.5 -dNoOutputFonts -sOutputFile="$$outlined" "$$figure"; \
		mv "$$outlined" "$$figure"; \
		pdfinfo "$$figure" | awk -v f="$$figure" '/^Pages:/ { if ($$2 != 1) exit 1; print f ": " $$2 " page" }'; \
		if pdffonts "$$figure" | tail -n +3 | grep -q '[[:alnum:]]'; then \
			echo "embedded figure font remains in $$figure" >&2; exit 1; \
		fi; \
	done

verify-falsification: falsification
	@test -s "$(MAIN_DIR)/aber_lr_falsification_chain.pdf"
	@pdfinfo "$(MAIN_DIR)/aber_lr_falsification_chain.pdf" | awk '/^Pages:/ { if ($$2 != 1) exit 1; print "aber_lr_falsification_chain.pdf: " $$2 " page" }'
