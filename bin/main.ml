open Network.NeuralNetwork

let get_labels base_folder = Sys.readdir base_folder |> Array.to_list

let get_image_size (filepath : string) : int * int =
  match Stb_image.load filepath with
  | Ok img -> (img.width, img.height)
  | Error (`Msg err) -> failwith ("Failed to load image: " ^ err)

let image_to_input (img : Stb_image.int8 Stb_image.t) : float list =
  let data = img.data in
  let size = Bigarray.Array1.dim data in
  List.init size (fun i -> float_of_int (Bigarray.Array1.get data i) /. 255.0)

let load_training_data base_folder =
  let labels = get_labels base_folder in
  let label_map = List.mapi (fun i label -> (label, i)) labels in
  let dataset = ref [] in
  let input_size = ref 0 in
  List.iter
    (fun (label, idx) ->
      let folder = Filename.concat base_folder label in
      if Sys.file_exists folder && Sys.is_directory folder then
        let files = Sys.readdir folder in
        Array.iter
          (fun file ->
            let filepath = Filename.concat folder file in
            match Stb_image.load filepath with
            | Ok img ->
                if !input_size = 0 then input_size := img.width * img.height;
                if img.width * img.height <> !input_size then
                  failwith "Image size mismatch in dataset!";
                let input = image_to_input img in
                let output =
                  Array.init (List.length labels) (fun j ->
                      if j = idx then 1.0 else 0.0)
                  |> Array.to_list
                in
                dataset := (input, output) :: !dataset
            | Error (`Msg err) ->
                Printf.printf "Failed to load %s: %s\n" filepath err)
          files)
    label_map;
  (!dataset, label_map, !input_size)

let classify_image network label_map filepath =
  match Stb_image.load filepath with
  | Ok img -> (
      let input = image_to_input img in
      let output = NetworkImpl.forward network input in
      let predicted_class =
        List.mapi (fun i x -> (i, x)) output
        |> List.sort (fun a b -> compare (snd b) (snd a))
        |> List.hd |> fst
      in
      let predicted_label =
        List.assoc_opt predicted_class
          (List.map (fun (x, y) -> (y, x)) label_map)
      in
      match predicted_label with
      | Some label -> Printf.printf "Predicted class: %s\n" label
      | None -> Printf.printf "Unknown class\n")
  | Error (`Msg err) -> Printf.printf "Failed to load image: %s\n" err

let () =
  let base_folder = "data" in
  let dataset, label_map, input_size = load_training_data base_folder in
  let hidden_layer_size = 128 in
  let network =
    NetworkImpl.create [ input_size; hidden_layer_size; List.length label_map ]
  in
  Printf.printf
    "Network created with input size: %d and hidden layer size: %d\n" input_size
    hidden_layer_size;
  let trained_net =
    try NetworkImpl.train network dataset 100 0.01 with
    | Sys_error msg ->
        Printf.printf "System error: %s\n" msg;
        exit 1
    | Failure msg ->
        Printf.printf "Failure: %s\n" msg;
        exit 1
    | _ ->
        Printf.printf "Unknown error occurred during training.\n";
        exit 1
  in
  Printf.printf "Training complete.\n";
  print_endline "Enter image path:";
  let filepath = read_line () in
  classify_image trained_net label_map filepath
