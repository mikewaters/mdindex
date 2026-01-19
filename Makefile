TESTPATH = ''

# As a coding agent, you must use this target to run the differential test suite after making changes;
# no need to run the entire test suite
agent-test:
	pytest --testmon $(TESTPATH)

# As a coding agent, if I want to run all tests after a major change, I should use this target.
agent-regression-test:
	pytest -n auto $(TESTPATH)

.PHONY: agent-test agent-regression-test