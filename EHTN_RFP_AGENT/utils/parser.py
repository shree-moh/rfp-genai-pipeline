def parse_llm_output(llm_output):
    lines = [line for line in llm_output.strip().split('\n') if line and 'Requirement' not in line]
    data = []
    for line in lines:
        parts = [p.strip() for p in line.split('|') if p.strip()]
        if len(parts) == 3:
            data.append({
                "Requirement": parts[0],
                "Department": parts[1],
                "Confidence": parts[2]
            })
    return data
