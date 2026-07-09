def call_video_sampling(json_content):
    if "function" in json_content:
        start_time = json_content["function"]["arguments"]["start_time"]
        end_time = json_content["function"]["arguments"]["end_time"]
    else:
        start_time = json_content["arguments"]["start_time"]
        end_time = json_content["arguments"]["end_time"]
    return str((lambda m, s: int(m) * 60 + int(s))(*start_time.split(':')))+"; "+str((lambda m, s: int(m) * 60 + int(s))(*end_time.split(':')))