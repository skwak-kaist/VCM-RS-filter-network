# This file is covered by the license agreement found in the file "license.txt" in the root of this project.

scale_factor_id_mapping = {
      2 : 0,
      4 : 1,
      8 : 2,
      16 : 3,
    }

id_scale_factor_mapping = dict(reversed(x) for x in scale_factor_id_mapping.items())


resample_len_id_mapping = {
      2 : 0,
      3 : 1,
      4 : 2,
      5 : 3,
    }

id_resample_len_mapping = dict(reversed(x) for x in resample_len_id_mapping.items())

predict_len_id_mapping = {
      1 : 0,
      2 : 1,
      3 : 2,
      4 : 3,
    }

id_predict_len_mapping = dict(reversed(x) for x in predict_len_id_mapping.items())